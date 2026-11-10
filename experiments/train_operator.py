"""Trains the deterministic operator of a GPO (stage 1) on a benchmark.

    python experiments/train_operator.py --bench darcy --arch wnof --level 2 --modes 12 12 --wavelet db6 --aug d4 \
        --epochs 500 --batch 20 --h1 0.1 --out checkpoints/darcy_s0

Relative-L2 loss (plus an optional H1 term on finite differences), AdamW with a one-cycle schedule, gradient clipping
and an EMA of the weights. The EMA with the best validation error is kept and written to <out>/ckpt.pt with the
normalisers; <out>/resume.pt allows an interrupted run to continue.
"""
import argparse, copy, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from gpo.data import BENCHMARKS, load_split
from gpo.operators import build_operator

p = argparse.ArgumentParser()
p.add_argument("--bench", required=True, choices=list(BENCHMARKS))
p.add_argument("--out", required=True)
p.add_argument("--ntrain", type=int, default=None)
p.add_argument("--nval", type=int, default=None)
p.add_argument("--ntest", type=int, default=None)
p.add_argument("--arch", default="wno", choices=["wno", "wnof", "ewno", "fno"])
p.add_argument("--width", type=int, default=64)
p.add_argument("--level", type=int, default=2)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--wavelet", default="db6")
p.add_argument("--modes", type=int, nargs="+", default=[16, 16])
p.add_argument("--pad", type=int, default=0)
p.add_argument("--no-fine", action="store_true")
p.add_argument("--epochs", type=int, default=500)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--h1", type=float, default=0.1, help="weight of the relative-L2 loss on first differences")
p.add_argument("--aug", default="none", choices=["none", "shift", "d4"],
               help="shift: periodic translations along the grid (1D); d4: rotations and reflections of a square grid")
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--log-every", type=int, default=10)
p.add_argument("--ckpt-every", type=int, default=50)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

b = BENCHMARKS[a.bench]
a.data = a.bench
a.ntrain, a.nval, a.ntest = a.ntrain or b["ntrain"], a.nval or b["nval"], a.ntest or b["ntest"]
a.tail_test = b["tail_test"]
torch.manual_seed(a.seed)
dev = torch.device(a.device)
(xtr, ytr), (xva, yva), (xte, yte) = load_split(a.bench, a.ntrain, a.nval, a.ntest, a.tail_test)
grid = tuple(xtr.shape[1:-1])


def d4(t, k):
    """k-th element (0..7) of the dihedral group acting on the grid axes of (b, s, s, c)."""
    t = t.transpose(1, 2) if k & 4 else t
    return torch.rot90(t, k & 3, (1, 2))


def d4_inv(t, k):
    t = torch.rot90(t, -(k & 3), (1, 2))
    return t.transpose(1, 2) if k & 4 else t


# Per-pixel normalisers, made invariant under the symmetry group used for augmentation.
xm, xs = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-8
ym, ys = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True) + 1e-8
if a.aug == "d4":
    xm, xs, ym, ys = (sum(d4(t, k) for k in range(8)) / 8 for t in (xm, xs, ym, ys))
elif a.aug == "shift":
    xm, xs, ym, ys = (v.expand_as(xm).clone() for v in (xtr.mean(), xtr.std() + 1e-8, ytr.mean(), ytr.std() + 1e-8))
xtr_n, xva_n, xte_n = (((x - xm) / xs).to(dev) for x in (xtr, xva, xte))
ytr, yva, yte, ym_d, ys_d = (t.to(dev) for t in (ytr, yva, yte, ym, ys))

model = build_operator(vars(a), grid, xtr.shape[-1]).to(dev)
ema = copy.deepcopy(model).requires_grad_(False)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps_per_ep = len(xtr_n) // a.batch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * steps_per_ep, pct_start=0.05)
rel = lambda p, t: (p - t).flatten(1).norm(dim=1) / t.flatten(1).norm(dim=1)
EB = int(max(1, min(64, 2 ** 31 // (8 * a.width * math.prod(grid)))))


def forward(x, average=True):
    """Decoded EMA prediction, averaged over D4 when that symmetry is used."""
    if a.aug != "d4" or not average:
        return ema(x) * ys_d + ym_d
    return sum(d4_inv(ema(d4(x, k)) * ys_d + ym_d, k) for k in range(8)) / 8


@torch.no_grad()
def score(x, y, average=True):
    ema.eval()
    return torch.cat([rel(forward(x[i:i + EB], average), y[i:i + EB]) for i in range(0, len(x), EB)]).mean().item()


def augment(xb, yb):
    if a.aug == "shift":
        s = int(torch.randint(xb.shape[1], (1,)))
        return xb.roll(s, 1), yb.roll(s, 1)
    if a.aug == "d4":
        k = int(torch.randint(8, (1,)))
        return d4(xb, k), d4(yb, k)
    return xb, yb


best, state, start, t0 = math.inf, None, 0, time.time()
resume = os.path.join(a.out, "resume.pt")
if os.path.exists(resume):
    R = torch.load(resume, weights_only=False, map_location=dev)
    model.load_state_dict(R["model"]); ema.load_state_dict(R["ema"])
    opt.load_state_dict(R["opt"]); sched.load_state_dict(R["sched"])
    start, best, state = R["ep"], R["best"], R["state"]
    torch.set_rng_state(R["rng"].cpu())
    if dev.type == "cuda":
        torch.cuda.set_rng_state(R["cuda_rng"].cpu(), dev)
    print(f"resumed at epoch {start}", flush=True)

for ep in range(start, a.epochs):
    model.train()
    perm = torch.randperm(len(xtr_n), device=dev)
    for i in range(steps_per_ep):
        idx = perm[i * a.batch:(i + 1) * a.batch]
        xb, yb = augment(xtr_n[idx], ytr[idx])
        out = model(xb) * ys_d + ym_d
        loss = rel(out, yb).mean()
        if a.h1 > 0:
            for dim in range(1, out.ndim - 1):
                loss = loss + a.h1 * rel(torch.diff(out, dim=dim), torch.diff(yb, dim=dim)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step(); sched.step()
        with torch.no_grad():
            d = min(0.999, (sched.last_epoch + 1) / (sched.last_epoch + 10))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)
    if (ep + 1) % a.log_every == 0 or ep == a.epochs - 1:
        v = score(xva_n, yva)
        if v < best:
            best, state = v, copy.deepcopy(ema.state_dict())
        print(f"ep {ep + 1} train {loss.item():.5f} val {100 * v:.3f}% t {time.time() - t0:.0f}s", flush=True)
    if (ep + 1) % a.ckpt_every == 0 and ep < a.epochs - 1:
        os.makedirs(a.out, exist_ok=True)
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "ep": ep + 1, "best": best, "state": state,
                    "rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.zeros(0)},
                   resume + ".tmp")
        os.replace(resume + ".tmp", resume)

ema.load_state_dict(state)
test, test_plain = score(xte_n, yte), score(xte_n, yte, average=False)
n_params = sum(q.numel() for q in model.parameters())
print(f"FINAL val {100 * best:.3f}% test {100 * test:.3f}% (no group averaging {100 * test_plain:.3f}%) "
      f"params {n_params}", flush=True)
os.makedirs(a.out, exist_ok=True)
torch.save({"state": state, "args": vars(a), "xm": xm, "xs": xs, "ym": ym, "ys": ys}, os.path.join(a.out, "ckpt.pt"))
json.dump({"val": best, "test": test, "test_plain": test_plain, "params": n_params, "time": time.time() - t0},
          open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
if os.path.exists(resume):
    os.remove(resume)
