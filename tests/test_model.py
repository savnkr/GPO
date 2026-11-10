import pytest
import torch

from gpo.model import GPO
from gpo.operators import WNO


def make(grid=(16,), n=12, seed=0, amplitude=False):
    torch.manual_seed(seed)
    op = WNO(grid, width=8, level=2, layers=2, wavelet="db4", proj=6).double()
    d = 1
    for g in grid:
        d *= g
    m = GPO(op, grid, embed_dim=3, out_scale=0.5 + torch.rand(d, dtype=torch.float64), amplitude=amplitude).double()
    with torch.no_grad():
        m.log_sa.fill_(-1.0); m.log_noise.fill_(-4.0); m.log_ell_x.fill_(-1.5)
        m.log_lam.fill_(-2.0); m.log_se.fill_(-2.5)
        if amplitude:
            m.log_amp[2].weight.normal_(0, 0.5); m.log_amp[2].bias.fill_(0.2)
    x = torch.randn(n + 3, *grid, 1, dtype=torch.float64)
    y = torch.randn(n, len(m.coords), dtype=torch.float64)
    return m, x[:n], y, x[n:]


def train_kernel(m, x):
    K = dense_kernel(m, x, x, same=True)
    g = m.features(x)[2].reshape(-1)
    K = K + m.jitter * m.sa2() * g[:, None] * torch.kron(torch.eye(len(x), dtype=K.dtype), m.kx()) * g[None]
    return K + m.noise() * torch.diag(g ** 2)


def dense_kernel(m, xa, xb, same=False):
    Pa, pa, ga = m.features(xa); Pb, pb, gb = m.features(xb)
    lin = torch.einsum("adk,bek->adbe", Pa, Pb)
    ka = m.ka(pa, pb) + (m.se2() * torch.eye(len(pa), dtype=pa.dtype) if same else 0)
    kron = ka[:, None, :, None] * m.kx()[None, :, None, :] * ga[:, :, None, None] * gb[None, None]
    return (lin + kron).reshape(Pa.shape[0] * Pa.shape[1], -1)


def dense_posterior(m, x, y, xs):
    with torch.no_grad():
        K = train_kernel(m, x)
        Ks, Kss = dense_kernel(m, xs, x), dense_kernel(m, xs, xs, same=True)
        mean = Ks @ torch.linalg.solve(K, y.reshape(-1))
        cov = Kss - Ks @ torch.linalg.solve(K, Ks.T)
    return mean.reshape(len(xs), -1), cov


@pytest.mark.parametrize("grid", [(16,), (6, 5)])
@pytest.mark.parametrize("amplitude", [False, True])
def test_posterior_mean_and_variance(grid, amplitude):
    m, x, y, xs = make(grid, amplitude=amplitude)
    m.fit(x, y)
    mu, var = m.predict(xs, batch=2)
    mu_d, cov_d = dense_posterior(m, x, y, xs)
    torch.testing.assert_close(mu, mu_d, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(var, cov_d.diagonal().reshape(var.shape), rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("amplitude", [False, True])
def test_nll_matches_dense(amplitude):
    m, x, y, _ = make(amplitude=amplitude)
    with torch.no_grad():
        K = train_kernel(m, x)
        ref = (0.5 * y.reshape(-1) @ torch.linalg.solve(K, y.reshape(-1)) + 0.5 * torch.logdet(K)) / y.numel()
        ref = ref + 0.5 * torch.log(torch.tensor(2 * torch.pi, dtype=K.dtype))
        torch.testing.assert_close(m.nll(x, y), ref)


def test_nll_backward_reaches_operator():
    m, x, y, _ = make()
    m.nll(x, y).backward()
    assert m.op.lift.weight.grad is not None and m.op.lift.weight.grad.abs().sum() > 0
    assert m.log_ell_x.grad is not None and torch.isfinite(m.log_ell_x.grad)


@pytest.mark.parametrize("amplitude", [False, True])
def test_matheron_samples_match_posterior(amplitude):
    """Pathwise-conditioned draws have the exact posterior mean and covariance."""
    m, x, y, xs = make(n=8, amplitude=amplitude)
    xs = xs[:2]
    m.fit(x, y)
    g = torch.Generator().manual_seed(0)
    S = m.sample(xs, n_samples=6000, batch=500, generator=g).reshape(6000, -1)
    mu_d, cov_d = dense_posterior(m, x, y, xs)
    emp_mu, emp_cov = S.mean(0), torch.cov(S.T)
    sd = cov_d.diagonal().sqrt()
    assert ((emp_mu - mu_d.reshape(-1)).abs() / sd).max() < 0.1
    corr_err = (emp_cov - cov_d) / (sd[:, None] * sd[None])
    assert corr_err.abs().max() < 0.1


@pytest.mark.parametrize("amplitude", [False, True])
def test_sensor_update_matches_joint_conditioning(amplitude):
    """Absorbing sensors after training equals conditioning on training fields and sensors jointly."""
    from gpo.conditioning import SensorUpdate, greedy_sensors
    m, x, y, xs = make(n=8, amplitude=amplitude)
    xs = xs[:1]
    m.fit(x, y)
    mu, var = m.predict(xs)
    J = [1, 5, 11]
    s = torch.randn(len(J), dtype=torch.float64)
    upd = SensorUpdate(m, xs, J)
    mu_new, var_new = upd.mean(mu[0], s), upd.variance(var[0])

    with torch.no_grad():
        Kxx = train_kernel(m, x)
        Ksx = dense_kernel(m, xs, x)
        Kss = dense_kernel(m, xs, xs, same=True)
        n = Kxx.shape[0]
        idx = torch.tensor(J)
        Kj = torch.cat([Ksx[idx], Kss[idx]], 1)
        Kall = torch.cat([torch.cat([Kxx, Ksx[idx].T], 1),
                          torch.cat([Ksx[idx], Kss[idx][:, idx] + m.noise() * torch.eye(len(J))], 1)])
        Kcross = torch.cat([Ksx, Kss[:, idx]], 1)
        obs = torch.cat([y.reshape(-1), s])
        mu_ref = Kcross @ torch.linalg.solve(Kall, obs)
        var_ref = Kss.diagonal() - (Kcross * torch.linalg.solve(Kall, Kcross.T).T).sum(1)
    torch.testing.assert_close(mu_new, mu_ref, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(var_new, var_ref, rtol=1e-5, atol=1e-8)

    J_g, hist = greedy_sensors(m, xs, 3)
    torch.testing.assert_close(hist[-1], SensorUpdate(m, xs, J_g).variance(var[0]), rtol=1e-6, atol=1e-9)



@pytest.mark.parametrize("amplitude", [False, True])
def test_paired_matheron_draws_match_posterior(amplitude):
    """Draws made from their own prior draws (return_prior) have the exact posterior law too."""
    m, x, y, xs = make(n=8, amplitude=amplitude)
    xs = xs[:2]
    m.fit(x, y)
    S, P = m.sample(xs, n_samples=6000, batch=500, generator=torch.Generator().manual_seed(1), return_prior=True)
    S = S.reshape(6000, -1)
    mu_d, cov_d = dense_posterior(m, x, y, xs)
    sd = cov_d.diagonal().sqrt()
    assert ((S.mean(0) - mu_d.reshape(-1)).abs() / sd).max() < 0.1
    assert ((torch.cov(S.T) - cov_d) / (sd[:, None] * sd[None])).abs().max() < 0.1


def test_greedy_sensors_leaves_candidates_untouched():
    from gpo.conditioning import greedy_sensors
    m, x, y, xs = make(n=8)
    m.fit(x, y)
    cand = torch.ones(len(m.coords), dtype=torch.bool)
    greedy_sensors(m, xs[:1], 3, candidates=cand)
    assert cand.all()
