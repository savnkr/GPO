"""Sensor fusion: a trained GPO absorbs m sparse readings of the unknown output field at test time.

Compares, per number of sensors m and over test fields:
  operator          the neural operator alone (no sensors)
  kriging           a spatial GP on the readings alone, prior mean and variance from the training fields
  operator+kriging  the operator corrected by kriging its residual at the sensors (stationary kernel)
  GPO (random)      exact GPO conditioning on randomly placed sensors
  GPO (greedy)      sensors placed where the GPO posterior variance is largest

    python experiments/sensor_fusion.py --run runs/gpo/navier_stokes_s0 --out results/apps/sensors_navier_stokes.pt
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from experiments.common import Run
from gpo.conditioning import SensorUpdate, greedy_sensors
from gpo.metrics import coverage

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--ntest", type=int, default=50)
p.add_argument("--m", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
p.add_argument("--frames", type=int, default=64)
p.add_argument("--sensor-noise", type=float, default=1e-6, help="measurement noise variance, model units")
a = p.parse_args()
torch.manual_seed(0)
R = Run(a.run)
m = R.model
d = len(m.coords)
dom = None
if R.bench == "darcy_notch":
    from gpo.data import notch_mask
    dom = notch_mask().flatten().to(R.Xte.device)
cand = dom if dom is not None else torch.ones(d, dtype=torch.bool, device=R.Xte.device)

Kx = m.cache["Kx"]
v_field = R.Ytr.var(0).mean()


def krige(prior_mean, prior_var, J, s, r):
    """Stationary GP regression with kernel prior_var * Kx on readings s at J."""
    G = prior_var * Kx[J][:, J] + r * torch.eye(len(J), dtype=Kx.dtype, device=Kx.device)
    w = torch.linalg.solve(G, s - prior_mean[J])
    return prior_mean + prior_var * Kx[:, J] @ w


rel = lambda u, v: ((R.decode(u) - R.decode(v)).norm() / R.decode(v).norm()).item()
r = torch.tensor(a.sensor_noise, dtype=torch.float64, device=R.Xte.device)
mean_field = R.Ytr.mean(0)
with torch.no_grad():
    op_all = lambda X: torch.cat([m.op(X[i:i + 25]) for i in range(0, len(X), 25)])
    ys_px = (R.s_out * m.out_scale).float().reshape(1, *R.grid, 1)
    ym_px = R.ym.float().reshape(1, *R.grid, 1)
    op_phys = lambda X: (op_all(X) * ys_px + ym_px).flatten(1).double()
    res_var = ((R.encode(op_phys(R.Xva)) - R.encode(R.yva)) ** 2).mean()
rows = []
frames = None
for i in range(min(a.ntest, len(R.Xte))):
    x1 = R.Xte[i:i + 1]
    y = R.encode(R.yte[i])
    mu, var = m.predict(x1)
    mu, var = mu[0], var[0]
    nz = m.noise_at(x1)[0]
    op = R.encode(op_phys(x1)[0])
    g = torch.Generator(device=x1.device).manual_seed(i)
    noise = r.sqrt() * torch.randn(d, generator=g, device=x1.device, dtype=y.dtype)
    obs = y + noise
    J_greedy, var_hist = greedy_sensors(m, x1, max(a.m), var=var, r=r, candidates=cand)
    pool = torch.nonzero(cand).flatten()
    J_rand_all = pool[torch.randperm(len(pool), generator=g, device=x1.device)[:max(a.m)]]
    row = {"operator": rel(op, y), "gpo": rel(mu, y), "m": a.m, "kriging": [], "op_kriging": [], "gpo_random": [],
           "gpo_greedy": [], "cov95_random": [], "cov95_greedy": [], "sd_random": [], "sd_greedy": []}
    for k in a.m:
        Jr, Jg = J_rand_all[:k], J_greedy[:k]
        row["kriging"].append(rel(krige(mean_field, v_field, Jr, obs[Jr], r), y))
        row["op_kriging"].append(rel(krige(op, res_var, Jr, obs[Jr], r), y))
        for name, J in (("random", Jr), ("greedy", Jg)):
            u = SensorUpdate(m, x1, J, r=r)
            mu_k, var_k = u.mean(mu, obs[J]), u.variance(var)
            sd = ((var_k + nz).sqrt() * R.tau)
            row[f"gpo_{name}"].append(rel(mu_k, y))
            row[f"cov95_{name}"].append(coverage(mu_k[None], sd[None], y[None], 0.95))
            row[f"sd_{name}"].append(sd.mean().item())
    rows.append(row)
    if i == 0:
        draws = m.sample(x1, n_samples=8, generator=g)[:, 0]
        frames = {"x": R.xte_raw[0].squeeze(-1), "truth": R.yte[0].float().cpu(), "operator": R.decode(op).float().cpu(),
                  "J": J_greedy[: a.frames].cpu(), "mean": [], "sd": [], "draws": [], "grid": R.grid}
        for k in range(a.frames + 1):
            J = J_greedy[:k]
            if k == 0:
                mu_k, var_k, dr = mu, var, draws
            else:
                u = SensorUpdate(m, x1, J, r=r)
                mu_k, var_k, dr = u.mean(mu, obs[J]), u.variance(var), u.draws(draws, obs[J], generator=g)
            frames["mean"].append(R.decode(mu_k).float().cpu())
            frames["sd"].append(((var_k + nz).sqrt() * R.s_out * R.tau).float().cpu())
            frames["draws"].append(R.decode(dr[:3]).float().cpu())
    print(i, {k: row[k] for k in ("operator", "gpo")}, flush=True)

summary = {k: torch.tensor([r_[k] for r_ in rows]).mean(0).tolist() for k in rows[0] if k != "m"}
print(summary, flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"bench": R.bench, "m": a.m, "rows": rows, "summary": summary, "frames": frames}, a.out)
