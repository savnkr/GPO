"""Numerical checks of the theory on a trained GPO: solver convergence, the inexact-Matheron bound, and
Monte-Carlo convergence of Matheron draws to the exact posterior covariance.

    python experiments/theory.py --run runs/gpo/burgers_s0 --out results/theory/burgers.pt
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from experiments.common import Run
from gpo.conditioning import covariance_columns
from gpo.solvers import KernelOperator, cg, sdd

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--ntest", type=int, default=20)
a = p.parse_args()
torch.manual_seed(0)
torch.set_grad_enabled(False)
R = Run(a.run)
m, c = R.model, R.model.cache
Ka = m._ka_train(c["p"])
K = KernelOperator(Ka, c["Kx"], m.noise(), c["Psi"])
Y, A_star = R.Ytr, c["A"]
Xs = R.Xte[: a.ntest]
mu_star, var_post = m.predict(Xs)
Psi_s, p_s, g_s = m.batched_features(Xs)
ka_s = m.ka(p_s, c["p"])
prior_var = (Psi_s ** 2).sum(-1) + g_s ** 2 * m.prior_a() * m.kx().diagonal()
explained = (prior_var - var_post).clamp_min(0)


def predict_from(A, w=None):
    if w is None:
        return m._cross(ka_s, Psi_s, g_s, A[None])[0]
    return Psi_s @ w + g_s * (ka_s @ (A @ c["Kx"]))


def dual_gap(A):
    D = A - A_star
    return 0.5 * (D * K.matvec(D)).sum().item()


def track(name, store):
    def cb(t, A, w=None):
        mu = predict_from(A, w)
        gap = dual_gap(A) if w is None else float("nan")
        err = (mu - mu_star).abs()
        bound = (2 * max(gap, 0)) ** 0.5 * explained.sqrt() if w is None else torch.full_like(explained, float("nan"))
        store.append({"t": t, "gap": gap, "max_err": err.max().item(), "rel_err": ((mu - mu_star).norm(dim=1)
                      / mu_star.norm(dim=1)).mean().item(), "bound_ratio": (err / bound.clamp_min(1e-30)).max().item(),
                      "residual": ((K.matvec(A) - Y).norm() / Y.norm()).item(), "time": time.time() - t0})
    return cb


out = {"bench": R.bench, "N": len(Y), "d": Y.shape[1], "c": c["Psi"].shape[-1]}
g = torch.Generator(device="cuda").manual_seed(0)
for name, kw in [("sdd_backfit", dict(lr=0.5, momentum=0.9, nesterov=True, backfit=50)),
                 ("sdd", dict(lr=0.5, momentum=0.9, nesterov=True)),
                 ("sdd_no_averaging", dict(lr=0.5, momentum=0.9, nesterov=False, averaging=1.0)),
                 ("sgd", dict(lr=0.5, momentum=0.0, nesterov=False, averaging=1.0))]:
    hist = []
    t0 = time.time()
    sdd(K, Y, steps=a.steps, batch=64, generator=g, callback=track(name, hist), every=max(1, a.steps // 100), **kw)
    out[name] = hist
    print(name, hist[-1], flush=True)
hist = []
t0 = time.time()
cg(K, Y, iters=300, tol=1e-12, callback=lambda t, A: track("cg", hist)(t, A) if t % 3 == 0 else None)
out["cg"] = hist
print("cg", hist[-1], flush=True)

x1 = R.Xte[:1]
d = Y.shape[1]
J = torch.arange(d, device=x1.device) if d <= 1024 else torch.randperm(d, device=x1.device)[:512]
C_exact = covariance_columns(m, x1, J)[J]
draws = m.sample(x1, n_samples=4096, batch=16, generator=g)[:, 0][:, J]
mc = []
for S in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
    Cs = torch.cov(draws[:S].T)
    mc.append({"S": S, "rel_fro": ((Cs - C_exact).norm() / C_exact.norm()).item(),
               "mean_err": ((draws[:S].mean(0) - mu_star[0][J]).norm() / C_exact.diagonal().sum().sqrt()).item()})
    print(mc[-1], flush=True)
out["matheron_mc"] = mc
out["cov_exact"] = C_exact.float().cpu()
out["cov_mc"] = torch.cov(draws[:256].T).float().cpu()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(out, a.out)
