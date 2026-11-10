import torch

from gpo.kernels import matern52
from gpo.linalg import KronEig, Woodbury
from gpo.solvers import KernelOperator, cg, sdd

torch.set_default_dtype(torch.float64)


def problem(N=40, d=12, c=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    za, zx = torch.randn(N, 3, generator=g), torch.rand(d, 1, generator=g)
    Ka, Kx = matern52(za, za, 1.0), matern52(zx, zx, 0.3)
    Psi = 0.2 * torch.randn(N, d, c, generator=g)
    R = torch.randn(N, d, generator=g)
    return Ka, Kx, torch.tensor(0.1), Psi, R


def exact(Ka, Kx, s2, Psi, R):
    return Woodbury(KronEig(Ka, Kx, s2), Psi).solve(R)


def test_matvec_matches_dense():
    Ka, Kx, s2, Psi, R = problem()
    K = KernelOperator(Ka, Kx, s2, Psi)
    F = Psi.reshape(-1, Psi.shape[-1])
    dense = torch.kron(Ka, Kx) + F @ F.T + s2 * torch.eye(F.shape[0])
    torch.testing.assert_close(K.matvec(R).reshape(-1), dense @ R.reshape(-1))


def test_cg_converges():
    Ka, Kx, s2, Psi, R = problem()
    X = cg(KernelOperator(Ka, Kx, s2, Psi), R, iters=2000, tol=1e-10)
    torch.testing.assert_close(X, exact(Ka, Kx, s2, Psi, R), rtol=1e-6, atol=1e-6)


def test_sdd_converges_and_tracks_products():
    Ka, Kx, s2, Psi, R = problem()
    K = KernelOperator(Ka, Kx, s2, Psi)
    X, PX, XK = sdd(K, R, steps=20000, batch=10, lr=0.5, generator=torch.Generator().manual_seed(0))
    X0 = exact(Ka, Kx, s2, Psi, R)
    assert (K.matvec(X) - R).norm() / R.norm() < 2e-2
    assert (K.matvec(X) - K.matvec(X0)).norm() / R.norm() < 2e-2
    torch.testing.assert_close(PX, K.psi_t(X))
    torch.testing.assert_close(XK, X @ Kx)


def test_sdd_multiple_rhs():
    Ka, Kx, s2, Psi, R = problem(N=20)
    K = KernelOperator(Ka, Kx, s2, Psi)
    Rs = torch.stack([R, 2 * R])
    X, _, _ = sdd(K, Rs, steps=4000, batch=5, generator=torch.Generator().manual_seed(1))
    torch.testing.assert_close(X[1], 2 * X[0])


def test_streaming_features_match_stored():
    Ka, Kx, s2, Psi, R = problem()
    K = KernelOperator(Ka, Kx, s2, Psi)
    Ks = KernelOperator(Ka, Kx, s2, lambda I: Psi[I], n_feat=Psi.shape[-1], chunk=7)
    torch.testing.assert_close(Ks.matvec(R), K.matvec(R))
    g = lambda: torch.Generator().manual_seed(3)
    X1, P1, _ = sdd(K, R, steps=300, batch=8, generator=g())
    X2, P2, _ = sdd(Ks, R, steps=300, batch=8, generator=g())
    torch.testing.assert_close(X1, X2)
    torch.testing.assert_close(P1, P2)


def test_backfit_sdd_matches_exact():
    Ka, Kx, s2, Psi, R = problem()
    Psi = 3.0 * Psi
    K = KernelOperator(Ka, Kx, s2, Psi)
    X0 = exact(Ka, Kx, s2, Psi, R)
    X, w, XK = sdd(K, R, steps=20000, batch=10, lr=0.5, backfit=20, generator=torch.Generator().manual_seed(0))
    assert (X - X0).norm() / X0.norm() < 2e-2
    torch.testing.assert_close(w, K.psi_t(X0), rtol=2e-2, atol=1e-3)
    torch.testing.assert_close(XK, X @ Kx)


def test_sdd_short_runs_stay_finite():
    Ka, Kx, s2, Psi, R = problem()
    X, _, _ = sdd(KernelOperator(Ka, Kx, s2, Psi), R, steps=30, batch=10, backfit=5,
                  generator=torch.Generator().manual_seed(0))
    assert torch.isfinite(X).all()
