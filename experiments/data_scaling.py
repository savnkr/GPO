"""Conditioning a trained GPO on more simulations than its operator was trained on, with stochastic dual descent.

The operator stays fixed; new fields enter only through the exact Gaussian process posterior, with no gradient
steps. The exact (Woodbury-Kronecker) solve stores N x d x c features and runs out of memory after a few thousand
fields; SDD recomputes features on demand and needs O(N d) memory. Optionally the kernel hyperparameters are first
re-estimated on fields the operator never saw, whose residuals reflect its generalisation error.

    python experiments/data_scaling.py --run runs/gpo/navier_stokes_s0 --extra 0 3000 9000 21000 33000 \
        --out results/apps/data_scaling_navier_stokes.pt
"""
import argparse, copy, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from experiments.common import Run
from gpo.data import navier_stokes
from gpo.solvers import KernelOperator, sdd

p = argparse.ArgumentParser()
p.add_argument("--run", required=True)
p.add_argument("--out", required=True)
p.add_argument("--extra", type=int, nargs="+", default=[0, 3000, 9000, 21000, 33000])
p.add_argument("--steps", type=int, default=6000)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--lr", type=float, default=0.5)
p.add_argument("--backfit", type=int, default=10)
p.add_argument("--refit-steps", type=int, default=300)
p.add_argument("--exact-max", type=int, default=6000)
a = p.parse_args()
torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
R = Run(a.run, fit=False)
m = R.model
assert R.bench == "navier_stokes"
n0 = R.metrics["init"]["ntrain"] + R.metrics["init"]["nval"] + R.metrics["init"]["ntest"]
x_all, y_all = navier_stokes(n0 + max(a.extra))
X_ex = R.encode_x(x_all[n0:])
Y_ex = R.encode(y_all[n0:].to(dev).flatten(1).double())
del x_all, y_all


@torch.no_grad()
def evaluate(model, X, Y, Xs):
    """SDD posterior mean at Xs given (X, Y), with features streamed from the operator."""
    Psi_te, p_te, _ = model.batched_features(Xs)
    p = torch.cat([model.batched_features(X[i:i + 500])[1] for i in range(0, len(X), 500)])
    Ka = model._ka_train(p)
    feat = lambda I: model.features(X[I])[0]
    K = KernelOperator(Ka, model.kx(), model.noise(), feat, n_feat=model.n_feat)
    steps = a.steps
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    _, PX, XK = sdd(K, Y, steps=steps, batch=a.batch, lr=a.lr, backfit=a.backfit,
                    generator=torch.Generator(device=dev).manual_seed(0))
    torch.cuda.synchronize()
    t = time.time() - t0
    mu = Psi_te @ PX + model.ka(p_te, p) @ XK
    return mu, t, torch.cuda.max_memory_allocated() / 2 ** 30


@torch.no_grad()
def exact(model, X, Y, Xs):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
    model.fit(X, Y)
    mu, _ = model.predict(Xs, var=False)
    torch.cuda.synchronize()
    t = time.time() - t0
    model.cache = None
    return mu, t, torch.cuda.max_memory_allocated() / 2 ** 30


rel = lambda mu: ((R.decode(mu) - R.yte).norm(dim=1) / R.yte.norm(dim=1)).mean().item()
models = {"trained": m}
if a.refit_steps and max(a.extra) > 0:
    m2 = copy.deepcopy(m)
    hyper = [q for n, q in m2.named_parameters() if n.startswith("log_")]
    for q in hyper:
        q.requires_grad_(True)
    opt = torch.optim.Adam(hyper, lr=3e-3)
    pool = max(a.extra)
    for step in range(a.refit_steps):
        idx = torch.randperm(pool, device=dev)[:256]
        loss = m2.nll(X_ex[idx], Y_ex[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    m2.eval().requires_grad_(False)
    print(f"refit on unseen fields: noise {m.noise().item():.2e} -> {m2.noise().item():.2e}, "
          f"sa2 {m.sa2().item():.2e} -> {m2.sa2().item():.2e}", flush=True)
    models["refit"] = m2

rows = []
for n_extra in a.extra:
    X = torch.cat([R.Xtr, X_ex[:n_extra]])
    Y = torch.cat([R.Ytr, Y_ex[:n_extra]])
    for name, model in models.items():
        row = {"N": len(X), "hypers": name}
        mu, t, mem = evaluate(model, X, Y, R.Xte)
        row.update(sdd_rl2=rel(mu), sdd_time=t, sdd_mem=mem, sdd_steps=a.steps)
        if len(X) <= a.exact_max:
            try:
                mu_e, t_e, mem_e = exact(model, X, Y, R.Xte)
                row.update(exact_rl2=rel(mu_e), exact_time=t_e, exact_mem=mem_e)
            except torch.OutOfMemoryError:
                row.update(exact_rl2=None, exact_oom=True)
                torch.cuda.empty_cache()
        rows.append(row)
        print(row, flush=True)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"rows": rows, "operator_test": R.metrics["operator_test"]}, a.out)
