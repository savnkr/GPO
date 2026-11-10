"""Uncertainty of derived quantities from joint posterior draws.

A probabilistic operator that only reports pointwise variances can draw fields with the right marginals but
independent pixels; any nonlinear or integrated quantity of such draws is miscalibrated. Matheron draws from the GPO
are joint, so intervals for derived quantities reflect how errors at different points move together. Reports 90% interval coverage, width and CRPS of
each quantity over the test fields, for joint draws and for independent draws with identical marginals.

    python experiments/functionals.py --run runs/gpo/navier_stokes_s0 --out results/apps/functionals_navier_stokes.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from experiments.common import Run

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--samples", type=int, default=256)
p.add_argument("--level", type=float, default=0.9)
a = p.parse_args()
R = Run(a.run)
m = R.model
grid = R.grid


def grad_energy(u):
    u = u.reshape(*u.shape[:-1], *grid)
    gs = [torch.diff(u, dim=-k) * grid[-k] for k in range(1, len(grid) + 1)]
    return sum((g ** 2).flatten(-len(grid)).mean(-1) for g in gs)


QOIS = {"mean": lambda u: u.mean(-1), "energy": lambda u: (u ** 2).mean(-1), "max": lambda u: u.amax(-1),
        "min": lambda u: u.amin(-1), "gradient energy": grad_energy}
if R.bench == "darcy_notch":
    from gpo.data import notch_mask
    dom = notch_mask().flatten().to(R.Xte.device)
    QOIS = {k: (lambda f: lambda u: f(torch.where(dom, u, torch.zeros_like(u))))(f) if k != "gradient energy" else f
            for k, f in QOIS.items()}

g = torch.Generator(device=R.Xte.device).manual_seed(0)
mu, var = m.predict(R.Xte)
draws = m.sample(R.Xte, n_samples=a.samples, generator=g)
noise = m.noise_at(R.Xte).sqrt()
joint = R.decode(mu + R.tau * (draws - mu + noise * torch.randn(draws.shape, generator=g, device=draws.device,
                                                                  dtype=draws.dtype)))
sd = (var + m.noise_at(R.Xte)).sqrt() * R.tau
indep = R.decode(mu + sd * torch.randn(draws.shape, generator=g, device=draws.device, dtype=draws.dtype))
truth = R.yte
lo, hi = (1 - a.level) / 2, (1 + a.level) / 2


def crps(s, y):
    return ((s - y).abs().mean(0) - 0.5 * (s[:, None] - s[None]).abs().mean((0, 1))).mean().item()


out = {"bench": R.bench, "level": a.level, "qoi": {}}
for name, f in QOIS.items():
    t = f(truth)
    if t.std() == 0:
        continue
    row = {}
    for kind, D in (("joint", joint), ("independent", indep)):
        q = f(D)
        ql, qh = q.quantile(lo, 0), q.quantile(hi, 0)
        row[kind] = {"coverage": ((t >= ql) & (t <= qh)).float().mean().item(),
                     "width": ((qh - ql) / t.std()).mean().item(), "crps": crps(q, t) / t.std().item()}
    out["qoi"][name] = row
    print(f"{name:16s} joint cov {row['joint']['coverage']:.2f} width {row['joint']['width']:.3g} | "
          f"independent cov {row['independent']['coverage']:.2f} width {row['independent']['width']:.3g}", flush=True)
pixel = ((truth >= joint.quantile(lo, 0)) & (truth <= joint.quantile(hi, 0))).float().mean().item()
out["pixel_coverage_joint"] = pixel
out["example"] = {"truth": truth[0].float().cpu(), "joint": joint[:6, 0].float().cpu(), "independent": indep[:6, 0].float().cpu()}
print("pixelwise coverage (joint)", round(pixel, 3))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save(out, a.out)
