"""Input-resolution independence: a GPO trained on one discretisation, given test inputs on other grids.

Each test input is sampled on a coarser or finer grid than the training one, mapped to the training grid
(gpo.resample), and the output is predicted on the training grid. No retraining.

    python experiments/resolution.py --run runs/gpo/burgers_s0 --out results/apps/resolution_burgers.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
from experiments.common import Run
from gpo import data
from gpo.metrics import coverage, gaussian_nll, rel_l2
from gpo.resample import to_grid

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.set_grad_enabled(False)
R = Run(a.run)
m = R.model
A = R.metrics["init"]
b = R.bench
n0 = A["ntrain"] + A["nval"]
te = slice(n0, n0 + A["ntest"]) if not A["tail_test"] else slice(-A["ntest"], None)


def inputs_at(level):
    """Test inputs of this benchmark on grid `level`, and whether the domain is periodic."""
    if b == "burgers":
        return data.burgers(sub=level)[0][te], True
    if b == "darcy":
        return data.darcy(sub=level)[0][te], False
    if b == "darcy_notch":
        return data.darcy_notch(sub=level)[0][te], False
    if b in ("advection", "advection_width"):
        ic = 1 if b == "advection_width" else 2
        params = np.load(f"{data.GEN}/advection_ic{ic}_200/advection_ic{ic}.npz")["params"][te]
        x = np.linspace(0, 1, level, endpoint=False)
        if ic == 1:
            f = lambda q: q[0] * (np.abs(np.mod(x - q[2] + 0.5, 1.0) - 0.5) <= q[1] / 2)
        else:
            def f(q):
                c1, w, h1, c2, s, h2 = q
                u1 = h1 * np.heaviside(x - (c1 - w / 2), 0.5) * np.heaviside(c1 + w / 2 - x, 0.5)
                return u1 + np.sqrt(np.maximum(h2 ** 2 - s ** 2 * (x - c2) ** 2, 0))
        return torch.from_numpy(np.stack([f(q) for q in params]).astype(np.float32))[..., None], True
    if b == "navier_stokes":
        return R.xte_raw if level == 64 else to_grid(R.xte_raw, (level, level), periodic=True), True
    raise ValueError(b)


LEVELS = {"burgers": [32, 16, 8, 4, 2, 1], "darcy": [10, 7, 5, 3, 2, 1], "darcy_notch": [5, 4, 2, 1],
          "advection": [100, 150, 200, 400, 800], "advection_width": [100, 150, 200, 400, 800],
          "navier_stokes": [16, 32, 48, 64]}
SMOOTH = {"burgers": True, "darcy": False, "darcy_notch": True, "advection": False, "advection_width": False,
          "navier_stokes": True}
rows, examples = [], {}
for level in LEVELS[b]:
    x, periodic = inputs_at(level)
    X = R.encode_x(to_grid(x, R.grid, periodic, SMOOTH[b]))
    mu, var = m.predict(X)
    mean = R.decode(mu)
    sd = R.sd(var, X)
    res = tuple(x.shape[1:-1])
    row = {"resolution": res, "points": int(np.prod(res)), "rl2": 100 * rel_l2(mean, R.yte),
           "cov95": coverage(mean, sd, R.yte, 0.95), "nll": gaussian_nll(mean, sd, R.yte),
           "train": res == tuple(R.grid)}
    rows.append(row)
    examples[res] = {"input": x[0].squeeze(-1).numpy(), "mean": mean[0].float().cpu().numpy(), "sd": sd[0].float().cpu().numpy()}
    print(row, flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"bench": b, "grid": R.grid, "rows": rows, "examples": examples, "truth": R.yte[0].float().cpu().numpy()}, a.out)
