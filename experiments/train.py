"""Trains a GPO on a benchmark from a pretrained operator feature map and evaluates it on the test split.

    python experiments/train.py --bench navier_stokes --init checkpoints/navier_stokes_s0 --out runs/navier_stokes_s0

Hyperparameters, the embedding and (optionally) the operator are learned on the exact marginal likelihood of random
subsets of training fields; the posterior conditions on all of them. The step with the best validation error is kept.
"""
import argparse, copy, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from gpo.data import domain_mask, load_split
from gpo.metrics import rel_l2, scorecard
from gpo.model import GPO
from gpo.operators import build_operator

p = argparse.ArgumentParser()
p.add_argument("--bench", required=True)
p.add_argument("--init", required=True)
p.add_argument("--out", required=True)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--subset", type=int, default=512)
p.add_argument("--lr", type=float, default=1e-2)
p.add_argument("--lr-op", type=float, default=0.0)
p.add_argument("--eval-every", type=int, default=50)
p.add_argument("--samples", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--heldout-steps", type=int, default=300,
               help="then re-estimate the kernel hyperparameters on the evidence of held-out fields (0: skip)")
p.add_argument("--heldout-cond", type=int, default=512)
p.add_argument("--no-linear", action="store_true", help="drop the operator-feature kernel")
p.add_argument("--no-spatial", action="store_true", help="independent output points (k_x = identity)")
p.add_argument("--amplitude", action="store_true",
               help="input- and location-dependent amplitude of the Kronecker part, learned on held-out evidence")
p.add_argument("--lr-amp", type=float, default=1e-3)
a = p.parse_args()
torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"

C = torch.load(os.path.join(a.init, "ckpt.pt"), weights_only=False, map_location="cpu")
A = C["args"]
(xtr, ytr), (xva, yva), (xte, yte) = load_split(a.bench, A["ntrain"], A["nval"], A["ntest"], tail_test=A["tail_test"])
grid = tuple(xtr.shape[1:-1])
op = build_operator(A, grid, xtr.shape[-1])
op.load_state_dict(C["state"])
xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
s_out = ys.mean()
model = GPO(op, grid, out_scale=(ys / s_out).expand(1, *grid, 1).flatten(), linear=not a.no_linear,
            spatial=not a.no_spatial, amplitude=a.amplitude).to(dev)
ym_f = ym.expand(1, *grid, 1).flatten().double()
enc_x = lambda x: (x.to(dev) - xm) / xs
enc_y = lambda y: ((y.to(dev).flatten(1).double() - ym_f) / s_out)
dec = lambda z: z.double() * s_out + ym_f
scale = s_out.double()
Xtr, Ytr, Xva, Xte = enc_x(xtr), enc_y(ytr), enc_x(xva), enc_x(xte)
Yva, Yte = yva.to(dev).flatten(1).double(), yte.to(dev).flatten(1).double()
mask = domain_mask(a.bench)

with torch.no_grad():
    op_dec = lambda X: torch.cat([op(X[i:i + 50]) for i in range(0, len(X), 50)]).mul(ys).add(ym).flatten(1).double()
    op_val, op_test = rel_l2(op_dec(Xva), Yva), rel_l2(op_dec(Xte), Yte)
print(f"operator alone: val {100 * op_val:.3f}%  test {100 * op_test:.3f}%", flush=True)

gp_params = [q for n, q in model.named_parameters() if not n.startswith(("op.", "log_amp."))]
groups = [{"params": gp_params, "lr": a.lr}]
if a.lr_op > 0:
    groups.append({"params": list(model.op.parameters()), "lr": a.lr_op})
else:
    model.op.requires_grad_(False)
opt = torch.optim.Adam(groups)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)


def validate():
    model.eval().fit(Xtr, Ytr)
    mu, _ = model.predict(Xva, var=False)
    return rel_l2(dec(mu), Yva)


history, best = [], (validate(), copy.deepcopy(model.state_dict()), 0)
print(f"step 0 val {100 * best[0]:.4f}%", flush=True)
torch.cuda.reset_peak_memory_stats()
t0, t_train = time.time(), 0.0
for step in range(1, a.steps + 1):
    model.train()
    te = time.time()
    idx = torch.randperm(len(Xtr), device=dev)[: a.subset]
    loss = model.nll(Xtr[idx], Ytr[idx])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(); sched.step()
    torch.cuda.synchronize(); t_train += time.time() - te
    if step % a.eval_every == 0 or step == a.steps:
        model.cache = None
        v = validate()
        history.append({"step": step, "nll": loss.item(), "val": v, "noise": model.noise().item(),
                        "sa2": model.sa2().item(), "ell_x": model.log_ell_x.exp().item()})
        if v < best[0]:
            best = (v, copy.deepcopy(model.state_dict()), step)
        print(f"step {step} nll {loss.item():.4f} val {100 * v:.4f}% noise {model.noise().item():.2e} "
              f"sa2 {model.sa2().item():.2e} ell_x {model.log_ell_x.exp().item():.3f} t {time.time() - t0:.0f}s", flush=True)

model.load_state_dict(best[1])
model.eval()
half = len(Xva) // 2
Yva_n = enc_y(yva)


def heldout_fit(model, steps, check=25):
    """Kernel hyperparameters (operator and embedding fixed) by -log p(y_val1 | y_train subset); the state with the
    best evidence of the other validation half is kept."""
    hyp = [q for n, q in model.named_parameters() if n.startswith("log_") and not n.startswith("log_amp.")]
    amp = [q for n, q in model.named_parameters() if n.startswith("log_amp.")]
    opt = torch.optim.Adam([{"params": hyp, "lr": a.lr}] + ([{"params": amp, "lr": a.lr_amp}] if amp else []))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    model.requires_grad_(False)
    for q in hyp + amp:
        q.requires_grad_(True)
    ic_check = torch.randperm(len(Xtr), device=dev, generator=torch.Generator(device=dev).manual_seed(a.seed))[: a.heldout_cond]
    ih_check = torch.arange(half, len(Xva), device=dev)[:64]

    def check_nll():
        with torch.no_grad():
            return model.conditional_nll(Xtr[ic_check], Ytr[ic_check], Xva[ih_check], Yva_n[ih_check]).item()

    best = (check_nll(), copy.deepcopy(model.state_dict()))
    for step in range(1, steps + 1):
        ic = torch.randperm(len(Xtr), device=dev)[: a.heldout_cond]
        ih = torch.randperm(half, device=dev)[: min(128, half)]
        loss = model.conditional_nll(Xtr[ic], Ytr[ic], Xva[ih], Yva_n[ih])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if step % check == 0 or step == steps:
            v = check_nll()
            if v < best[0]:
                best = (v, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])
    model.requires_grad_(False)
    return model.eval()


def evaluate(model):
    tf = time.time()
    model.fit(Xtr, Ytr)
    torch.cuda.synchronize(); t_fit = time.time() - tf
    mu_v, var_v = model.predict(Xva)
    mu_t, var_t = model.predict(Xte)
    sd = lambda v, X: (v + model.noise_at(X)).sqrt() * scale
    mv, sv, mt, st = dec(mu_v), sd(var_v, Xva), dec(mu_t), sd(var_t, Xte)
    tau = ((Yva[half:] - mv[half:]) ** 2 / sv[half:] ** 2).mean().sqrt().item()
    g = torch.Generator(device=dev).manual_seed(a.seed)
    draws = model.sample(Xte, n_samples=a.samples, generator=g)
    draws_y = dec(draws) + model.noise_at(Xte).sqrt() * scale * torch.randn(draws.shape, generator=g, device=dev, dtype=draws.dtype)
    out = {"gpo_test": rel_l2(mt, Yte), "gpo_val": rel_l2(mv, Yva), "uq_raw": scorecard(mt, st, Yte, draws_y),
           "uq_calibrated": scorecard(mt, st * tau, Yte), "tau": tau,
           "hyper": {"noise": model.noise().item(), "sa2": model.sa2().item(), "ell_x": model.log_ell_x.exp().item(),
                     "latent_fraction": (var_t.mean() / (var_t.mean() + model.noise_at(Xte).mean())).item()},
           "time_fit": t_fit}
    if mask is not None:
        out["gpo_test_domain"] = rel_l2(mt[:, mask.to(dev)], Yte[:, mask.to(dev)])
    pred = {"mean": mt.float().cpu(), "sd": st.float().cpu(), "tau": tau, "draws": draws_y[:16].float().cpu()}
    model.cache = None
    return out, pred


res_mll, pred_mll = evaluate(model)
state_mll = copy.deepcopy(model.state_dict())
res = {"bench": a.bench, "operator_test": op_test, "operator_val": op_val, "best_step": best[2], "mll": res_mll,
       "time_per_step": t_train / a.steps, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
       "history": history, "args": vars(a), "init": {k: v for k, v in A.items() if isinstance(v, (int, float, str, list, bool))}}
os.makedirs(a.out, exist_ok=True)
torch.save(state_mll, os.path.join(a.out, "model_mll.pt"))
torch.save(pred_mll, os.path.join(a.out, "pred_mll.pt"))
final, pred = res_mll, pred_mll
if a.heldout_steps:
    th = time.time()
    heldout_fit(model, a.heldout_steps)
    res["time_heldout"] = time.time() - th
    final, pred = evaluate(model)
    res["heldout"] = final
res.update({k: final[k] for k in ("gpo_test", "uq_raw", "uq_calibrated", "tau", "hyper", "gpo_test_domain") if k in final})
json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
torch.save(pred, os.path.join(a.out, "pred.pt"))
print(f"FINAL test rL2: operator {100 * op_test:.4f}%  GPO(mll) {100 * res_mll['gpo_test']:.4f}%  "
      f"GPO {100 * res['gpo_test']:.4f}%  cov95 raw {res['uq_raw']['cov95']:.3f} cal {res['uq_calibrated']['cov95']:.3f}  "
      f"nll {res['uq_raw']['nll']:.3f}  rho_field {res['uq_raw']['rho_field']:.2f}  mem {res['peak_mem_gb']:.1f} GB", flush=True)
