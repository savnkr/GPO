"""Loads a trained GPO run (experiments/train.py output) with its data in model units."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from gpo.data import load_split
from gpo.model import GPO
from gpo.operators import build_operator


class Run:
    def __init__(self, run, dev="cuda" if torch.cuda.is_available() else "cpu", fit=True, ntrain=None, strict=True):
        self.metrics = json.load(open(os.path.join(run, "metrics.json")))
        a = self.metrics["args"]
        C = torch.load(os.path.join(a["init"], "ckpt.pt"), weights_only=False, map_location="cpu")
        A = C["args"]
        self.bench = a["bench"]
        (xtr, ytr), (xva, yva), (xte, yte) = load_split(a["bench"], A["ntrain"], A["nval"], A["ntest"],
                                                         tail_test=A["tail_test"])
        self.grid = tuple(xtr.shape[1:-1])
        xm, xs, ym, ys = (C[k].to(dev) for k in ("xm", "xs", "ym", "ys"))
        self.s_out = ys.mean().double()
        self.ym = ym.expand(1, *self.grid, 1).flatten().double()
        op = build_operator(A, self.grid, xtr.shape[-1])
        self.model = GPO(op, self.grid, out_scale=(ys / ys.mean()).expand(1, *self.grid, 1).flatten(),
                         linear=not a.get("no_linear", False), spatial=not a.get("no_spatial", False),
                         amplitude=a.get("amplitude", False)).to(dev)
        self.model.load_state_dict(torch.load(os.path.join(run, "model.pt"), map_location=dev), strict=strict)
        self.model.eval().requires_grad_(False)
        enc = lambda x: (x.to(dev) - xm) / xs
        self.encode_x = enc
        self.Xtr, self.Xva, self.Xte = enc(xtr), enc(xva), enc(xte)
        self.Ytr = self.encode(ytr.to(dev).flatten(1).double())
        self.ytr, self.yva, self.yte = (t.to(dev).flatten(1).double() for t in (ytr, yva, yte))
        self.xte_raw = xte
        self.tau = self.metrics["tau"]
        if ntrain:
            self.Xtr, self.Ytr, self.ytr = self.Xtr[:ntrain], self.Ytr[:ntrain], self.ytr[:ntrain]
        if fit:
            self.model.fit(self.Xtr, self.Ytr)

    def encode(self, y):
        return (y - self.ym) / self.s_out

    def decode(self, z):
        return z * self.s_out + self.ym

    def sd(self, var, x, calibrated=True):
        """Predictive std of the observed fields at inputs x in physical units, from their latent variance."""
        return (var + self.model.noise_at(x)).sqrt() * self.s_out * (self.tau if calibrated else 1.0)
