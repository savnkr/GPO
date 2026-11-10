"""Accuracy and proper scores for field-valued predictions of shape (b, d)."""
import math
import torch


def rel_l2(m, y):
    return ((m - y).norm(dim=1) / y.norm(dim=1)).mean().item()


def gaussian_nll(m, s, y):
    return (0.5 * ((y - m) / s) ** 2 + s.log() + 0.5 * math.log(2 * math.pi)).mean().item()


def gaussian_crps(m, s, y):
    z = (y - m) / s
    n = torch.distributions.Normal(0.0, 1.0)
    return (s * (z * (2 * n.cdf(z) - 1) + 2 * n.log_prob(z).exp() - 1 / math.sqrt(math.pi))).mean().item()


def coverage(m, s, y, level):
    q = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.5 + level / 2, dtype=m.dtype))
    return ((y - m).abs() <= q * s).float().mean().item()


def spearman(a, b):
    r = lambda v: v.argsort().argsort().double()
    return torch.corrcoef(torch.stack([r(a), r(b)]))[0, 1].item()


def energy_score(samples, y):
    """samples (S, b, d); normalised by sqrt(d) so that grids are comparable."""
    t1 = (samples - y).norm(dim=-1).mean(0)
    t2 = (samples[:-1] - samples[1:]).norm(dim=-1).mean(0)
    return ((t1 - 0.5 * t2) / y.shape[-1] ** 0.5).mean().item()


def scorecard(m, s, y, samples=None):
    a, b = s.flatten(), (y - m).abs().flatten()
    if len(a) > 2_000_000:
        i = torch.randperm(len(a), device=a.device)[:2_000_000]
        a, b = a[i], b[i]
    out = {"rl2_pct": 100 * rel_l2(m, y), "nll": gaussian_nll(m, s, y), "crps": gaussian_crps(m, s, y),
           "cov50": coverage(m, s, y, 0.5), "cov90": coverage(m, s, y, 0.9), "cov95": coverage(m, s, y, 0.95),
           "sharpness": s.mean().item(), "rho_point": spearman(a, b),
           "rho_field": spearman(s.pow(2).mean(1), (y - m).pow(2).mean(1))}
    if samples is not None:
        out["energy"] = energy_score(samples, y)
    return out
