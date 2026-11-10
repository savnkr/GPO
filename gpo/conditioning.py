"""Test-time conditioning on sparse measurements of the output field (sensor fusion).

After a GPO has conditioned on the training fields, a few noisy readings s = u*(x_J) + e of the unknown output for a
new input a* are absorbed exactly. With C the posterior covariance of f(a*, .) given the training data,

    mean <- mean + C(., J) (C_JJ + r I)^-1 (s - mean_J),
    f    <- f    + C(., J) (C_JJ + r I)^-1 (s - f_J - e),     e ~ N(0, r I)    (Matheron's rule, per posterior draw),

so the joint draws from `GPO.sample` become draws conditioned on both the training set and the sensors.
"""
import torch


@torch.no_grad()
def covariance_columns(model, x_star, J):
    """Posterior covariance C(., J) (d, m) of the latent field at one input x_star (1, *grid, 1), given the training set."""
    c = model.cache
    Psi_s, p_s, g_s = model.features(x_star)
    U, gamma = model._factors(Psi_s, model.ka(p_s, c["p"]), g_s)
    U, gamma, g, Qx = U[0], gamma[0], g_s[0], c["kron"].Qx
    J = torch.as_tensor(J, device=Psi_s.device)
    return U.T @ U[:, J] + g[:, None] * ((Qx * gamma) @ Qx[J].T) * g[J][None]


class SensorUpdate:
    """Exact Gaussian update of one predicted field on readings at grid indices J with noise variance r."""

    def __init__(self, model, x_star, J, r=None):
        self.J = torch.as_tensor(J, device=x_star.device)
        self.r = model.noise() if r is None else torch.as_tensor(r, dtype=torch.float64, device=x_star.device)
        self.C = covariance_columns(model, x_star, self.J)
        G = self.C[self.J]
        G = 0.5 * (G + G.T) + (self.r + 1e-10 * G.diagonal().mean()) * torch.eye(len(self.J), dtype=G.dtype, device=G.device)
        self.L = torch.linalg.cholesky(G)

    def gain(self, innovation):
        return self.C @ torch.cholesky_solve(innovation.T, self.L)

    def mean(self, mean, s):
        return mean + self.gain((s - mean[self.J])[None]).T[0]

    def variance(self, var):
        W = torch.cholesky_solve(self.C.T, self.L)
        return (var - (self.C * W.T).sum(1)).clamp_min(0)

    def draws(self, draws, s, generator=None):
        """Matheron update of posterior draws (S, d)."""
        e = self.r.sqrt() * torch.randn(len(draws), len(self.J), dtype=draws.dtype, device=draws.device,
                                         generator=generator)
        return draws + self.gain(s[None] - draws[:, self.J] - e).T


@torch.no_grad()
def greedy_sensors(model, x_star, m, var=None, r=None, candidates=None):
    """Places m sensors one at a time where the current posterior variance is largest (pivoted Cholesky of C).

    Returns the indices and the variance after each placement, (m + 1, d).
    """
    if var is None:
        _, var = model.predict(x_star)
        var = var[0]
    r = model.noise() if r is None else torch.as_tensor(r, dtype=torch.float64, device=var.device)
    allowed = torch.ones_like(var, dtype=torch.bool) if candidates is None else candidates.to(var.device).clone()
    U, J, history = [], [], [var.clone()]
    for _ in range(m):
        j = torch.where(allowed, var, torch.full_like(var, -1.0)).argmax().item()
        col = covariance_columns(model, x_star, [j])[:, 0]
        for u in U:
            col = col - u * u[j]
        u = col / (col[j] + r).sqrt()
        U.append(u); J.append(j)
        var = (var - u ** 2).clamp_min(0)
        allowed[j] = False
        history.append(var.clone())
    return torch.tensor(J, device=var.device), torch.stack(history)
