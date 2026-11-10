"""Gaussian process operator with an operator-embedded kernel.

    f ~ GP(0, k),   k((a, x), (a', x')) = phi(a, x)^T Lambda phi(a', x') + [s_a^2 k_a(psi(a), psi(a')) + s_e^2 1(a = a')] k_x(x, x'),
    y = f + eps,    eps ~ N(0, s^2).

phi(a, x) are the pointwise features of a neural operator (its last hidden layer through the first projection layer,
plus a constant), psi(a) a learned embedding of the pooled features, and k_a, k_x Matern-5/2 kernels. The prior is
zero-mean: the operator enters only through the covariance. The linear part is a Bayesian projection head, so the
operator's own prediction is a posterior mean of a degenerate member of this family; the Kronecker part models
what the head cannot, with spatial correlation, which makes posterior draws physically meaningful fields. The s_e term
is spatially correlated error that does not transfer between inputs, the form an operator's generalisation error takes.
Optionally the Kronecker part and the noise are scaled by an amplitude g(a, x), so that the uncertainty can concentrate
where the operator errs, for instance at discontinuities or near boundaries; with K = G (S + Psi Psi^T / g^2) G the
inference stays exact.

Inference is exact (see `gpo.linalg`), posterior draws use Matheron's rule (pathwise conditioning).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import grid_coords, matern52
from .linalg import KronEig, Woodbury, eigh, exact_nll


class GPO(nn.Module):
    def __init__(self, operator, grid, embed_dim=32, jitter=1e-6, chunk=32, out_scale=None, linear=True, spatial=True,
                 amplitude=False):
        """linear=False drops the operator-feature kernel and spatial=False replaces k_x by the identity; with both
        off the kernel is the input-embedding kernel of the original GPO (outputs independent across grid points).
        amplitude=True scales the Kronecker part and the noise by g(a, x) = exp(3 tanh(r(h(a, x), x) / 3)), r a small
        network of the operator's pointwise features and the position, which makes the uncertainty vary over the field
        and between inputs."""
        super().__init__()
        self.linear, self.spatial = linear, spatial
        self.op, self.grid, self.jitter, self.chunk = operator, tuple(grid), jitter, chunk
        width = operator.width
        self.head = operator.proj[0]
        c = self.head.out_features + 1
        w2 = operator.proj[2].weight.detach().reshape(-1)
        b2 = operator.proj[2].bias.detach().reshape(-1)
        lam0 = torch.cat([w2 ** 2 + 1e-3 * (w2 ** 2).mean(), b2 ** 2 + 1.0])
        self.log_lam = nn.Parameter(lam0.log())
        self.embed = nn.Sequential(nn.Linear(2 * width, 128), nn.GELU(), nn.Linear(128, embed_dim))
        self.log_ell_a = nn.Parameter(torch.zeros(embed_dim))
        self.log_sa = nn.Parameter(torch.tensor(math.log(1e-2)))
        self.log_ell_x = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_noise = nn.Parameter(torch.tensor(math.log(1e-4)))
        self.log_se = nn.Parameter(torch.tensor(math.log(1e-3)))
        self.register_buffer("coords", grid_coords(self.grid))
        d = len(self.coords)
        self.register_buffer("out_scale", torch.ones(d, dtype=torch.float64) if out_scale is None
                             else out_scale.reshape(d).double())
        self.n_feat = c if linear else 1
        self.log_amp = None
        if amplitude:
            self.log_amp = nn.Sequential(nn.Linear(width + len(self.grid), 32), nn.GELU(), nn.Linear(32, 1))
            nn.init.zeros_(self.log_amp[2].weight); nn.init.zeros_(self.log_amp[2].bias)
        self.cache = None

    def noise(self):
        return self.log_noise.exp().double()

    def sa2(self):
        return (2 * self.log_sa).exp().double()

    def se2(self):
        return (2 * self.log_se).exp().double()

    def prior_a(self):
        """k_a(a, a) of a single input, including the per-input term."""
        return self.sa2() + self.se2()

    def features(self, x):
        """Scaled pointwise features Psi (b, d, c), input embedding (b, e) and amplitude g (b, d), float64."""
        h = self.op.features(x)
        f = F.gelu(self.head(h)).flatten(1, -2)
        f = torch.cat([f, torch.ones_like(f[..., :1])], -1)
        hf = h.flatten(1, -2)
        p = self.embed(torch.cat([hf.mean(1), hf.std(1)], -1))
        Psi = f.double() * (0.5 * self.log_lam.double()).exp() * self.out_scale[:, None]
        if not self.linear:
            Psi = torch.zeros_like(Psi[..., :1])
        if self.log_amp is None:
            g = torch.ones(Psi.shape[:2], dtype=torch.float64, device=Psi.device)
        else:
            xh = torch.cat([h, self.coords.to(h.dtype).reshape(1, *h.shape[1:-1], -1).expand(*h.shape[:-1], -1)], -1)
            g = (3 * torch.tanh(self.log_amp(xh).flatten(1).double() / 3)).exp()
        return Psi, p.double(), g

    def ka(self, p, q):
        ell = self.log_ell_a.exp().double()
        return self.sa2() * matern52(p / ell, q / ell)

    def kx(self):
        d = len(self.coords)
        if not self.spatial:
            return (1 + self.jitter) * torch.eye(d, dtype=torch.float64, device=self.coords.device)
        return matern52(self.coords, self.coords, self.log_ell_x.exp().double()) + self.jitter * torch.eye(
            d, dtype=torch.float64, device=self.coords.device)

    def _ka_train(self, p):
        return self.ka(p, p) + (self.jitter * self.sa2() + self.se2()) * torch.eye(len(p), dtype=p.dtype, device=p.device)

    def nll(self, x, y):
        """Exact negative log marginal likelihood per observation of the fields y (b, d) given inputs x."""
        Psi, p, g = self.features(x)
        n, d = y.shape
        L = exact_nll(self._ka_train(p), self.kx(), self.noise(), Psi / g[..., None], y.double() / g) + g.log().sum()
        return L / (n * d) + 0.5 * math.log(2 * math.pi)

    def conditional_nll(self, xc, yc, xh, yh):
        """-log p(y_h | y_c) per held-out value: the evidence of held-out fields given conditioning fields."""
        n_c, n_h, d = len(xc), len(xh), yh.shape[1]
        joint = self.nll(torch.cat([xc, xh]), torch.cat([yc, yh])) * (n_c + n_h)
        return (joint - self.nll(xc, yc) * n_c) / n_h

    @torch.no_grad()
    def batched_features(self, x, batch=50):
        out = [self.features(x[i:i + batch]) for i in range(0, len(x), batch)]
        return tuple(torch.cat([o[k] for o in out]) for k in range(3))

    @torch.no_grad()
    def fit(self, x, y):
        """Conditions on the training fields; caches everything predictions and draws need.

        The posterior is kept in blocked form: the feature weights w | y ~ N(w_hat, M^-1) with M = I + Psi^T S^-1 Psi,
        and the Kronecker part given w through S = Ka (x) Kx + s^2 I, whose eigenbasis makes every solve exact and
        well conditioned. Representer weights A = K^-1 y = S^-1 (y - Psi w_hat). With an amplitude g the same holds
        for y / g and Psi / g, since K = G (S + Psi Psi^T / g^2) G with G = diag(g).
        """
        Psi, p, g = self.batched_features(x)
        Kx = self.kx()
        kron = KronEig(self._ka_train(p), Kx, self.noise(), eig_x=eigh(Kx))
        wb = Woodbury(kron, Psi / g[..., None], chunk=self.chunk)
        y = y.double() / g
        w = torch.cholesky_solve((y.reshape(1, -1) @ Woodbury.flat(wb.P)).T, wb.Lm).reshape(-1)
        A = kron.solve(y - wb.Psi @ w)
        self.cache = dict(Psi=wb.Psi, p=p, Kx=Kx, kron=kron, wb=wb, A=A, w=w, AKx=A @ Kx, y=y)
        return self

    def _cross(self, ka_s, Psi_s, g_s, V):
        """k(test, train) V for V (..., N, d), training values divided by g: Psi_s (Psi^T V) + g_s Ka(test, train) V Kx."""
        c = self.cache
        N, d = V.shape[-2:]
        t = V.reshape(-1, N * d) @ Woodbury.flat(c["Psi"])
        lin = (Woodbury.flat(Psi_s) @ t.T).T.reshape(*V.shape[:-2], *Psi_s.shape[:2])
        return lin + g_s * torch.matmul(ka_s, V @ c["Kx"])

    def _factors(self, Psi_s, ka_s, g_s):
        """Posterior covariance factors at test inputs: C(x, x') = U(x)^T U(x') + g(x) g(x') sum_q Qx[x,q] Qx[x',q] gamma_q.

        U = Lm^-1 (Psi_s - Kx Ka(., X) P)^T (b, c, d) carries the uncertainty of the feature weights, gamma (b, d) the
        Kronecker part per spatial eigenvector; both are PSD by construction, so nothing cancels.
        """
        c = self.cache
        kron, wb, Kx = c["kron"], c["wb"], c["Kx"]
        b, d, k = Psi_s.shape
        B = Kx @ (ka_s @ wb.P.reshape(ka_s.shape[1], -1)).reshape(b, d, k)
        H = (Psi_s - g_s[..., None] * B).permute(2, 0, 1).reshape(k, -1)
        U = torch.linalg.solve_triangular(wb.Lm, H, upper=False).reshape(k, b, d).permute(1, 0, 2)
        a2 = (ka_s @ kron.Qa) ** 2
        gamma = kron.lx[None] * (self.prior_a() - kron.lx[None] * (a2 @ (1 / kron.lam)))
        return U, gamma.clamp_min(0)

    @torch.no_grad()
    def predict(self, x, batch=10, var=True, noise=False):
        """Posterior mean (b, d) and exact marginal variance (b, d) of the latent field, or of an observation with
        noise=True, normalised units."""
        c = self.cache
        Q2 = c["kron"].Qx ** 2
        means, variances = [], []
        for i in range(0, len(x), batch):
            Psi_s, p_s, g_s = self.features(x[i:i + batch])
            ka_s = self.ka(p_s, c["p"])
            means.append(Psi_s @ c["w"] + g_s * (ka_s @ c["AKx"]))
            if var:
                U, gamma = self._factors(Psi_s, ka_s, g_s)
                variances.append((U ** 2).sum(1) + g_s ** 2 * (gamma @ Q2.T + (self.noise() if noise else 0.0)))
        return torch.cat(means), (torch.cat(variances) if var else None)

    @torch.no_grad()
    def noise_at(self, x, batch=10):
        """Observation noise variance (b, d) at the grid points of each input."""
        g = torch.cat([self.features(x[i:i + batch])[2] for i in range(0, len(x), batch)])
        return self.noise() * g ** 2

    @torch.no_grad()
    def sample(self, x, n_samples=32, batch=8, generator=None, return_prior=False):
        """Joint posterior draws of the latent fields at the test inputs, (n_samples, b, d), by Matheron's rule.

        With f = Psi w + g, w ~ N(0, I) and g ~ GP(0, k_a k_x), the weights are drawn from their exact posterior
        w = w_hat + Lm^-T z, and the Kronecker part is conditioned pathwise given w:

            g | y, w  =  g + k_g(., X) S^-1 (y - Psi w - g(X) - eps),   (g, eps) drawn from the prior.

        With return_prior, each draw is instead made from its own prior draw f = Psi z + g by the plain rule
        f + k(., X) K^-1 (y - f(X) - eps), and both are returned: the same law, paired draw by draw. The blocked
        form is the default because it never cancels large terms; the plain form can lose accuracy when the
        feature block is very ill-conditioned.
        """
        c = self.cache
        Psi_s, p_s, g_s = self.batched_features(x)
        N, b = len(c["p"]), len(p_s)
        p_all = torch.cat([c["p"], p_s])
        la, Qa = eigh(self._ka_train(p_all))
        La = Qa * la.sqrt()
        kron, wb = c["kron"], c["wb"]
        Lx = kron.Qx * kron.lx.sqrt()
        ka_s = self.ka(p_s, c["p"])
        flat_tr, flat_te = Woodbury.flat(c["Psi"]), Woodbury.flat(Psi_s)
        out, prior = [], []
        dev, dt = p_s.device, torch.float64
        for s0 in range(0, n_samples, batch):
            s = min(batch, n_samples - s0)
            z = torch.randn(s, Psi_s.shape[-1], generator=generator, device=dev, dtype=dt)
            w = c["w"] + torch.linalg.solve_triangular(wb.Lm.T, z.T, upper=True).T
            g0 = La @ torch.randn(s, N + b, Lx.shape[0], generator=generator, device=dev, dtype=dt) @ Lx.T
            eps = self.noise().sqrt() * torch.randn(s, N, Lx.shape[0], generator=generator, device=dev, dtype=dt)
            if return_prior:
                f_tr = (flat_tr @ z.T).T.reshape(s, N, -1) + g0[:, :N]
                f_te = (flat_te @ z.T).T.reshape(s, b, -1) + g_s * g0[:, N:]
                out.append(f_te + self._cross(ka_s, Psi_s, g_s, wb.solve(c["y"] - f_tr - eps)))
                prior.append(f_te)
                continue
            lin_tr = (flat_tr @ w.T).T.reshape(s, N, -1)
            V = kron.solve(c["y"] - lin_tr - g0[:, :N] - eps)
            g = g0[:, N:] + torch.matmul(ka_s, V @ c["Kx"])
            out.append((flat_te @ w.T).T.reshape(s, b, -1) + g_s * g)
        return (torch.cat(out), torch.cat(prior)) if return_prior else torch.cat(out)
