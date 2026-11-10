"""Collects runs/*/<bench>_s<seed>/metrics.json into results/summary.json and the markdown tables of the README.

    python scripts/collect.py --runs runs/gpo --ablation runs/ablation
"""
import argparse, glob, json, os
import numpy as np

BENCH = ["burgers", "advection_width", "advection", "darcy", "darcy_notch", "navier_stokes"]
NAMES = {"burgers": "Burgers", "advection_width": "Wave advection, varying width", "advection": "Wave advection, wave and bump",
         "darcy": "Darcy flow", "darcy_notch": "Darcy, notched domain", "navier_stokes": "Navier–Stokes"}
PAPER = {"burgers": (2.89, 0.11), "advection_width": (0.63, 0.08), "darcy": (3.98, 0.12), "darcy_notch": (2.18, 0.19),
         "navier_stokes": (2.21, 0.44)}
PAPER_WNO = {"burgers": 4.09, "advection_width": 1.01, "darcy": 2.82, "darcy_notch": 2.21, "navier_stokes": 12.23}
PAPER_GP = {"burgers": 4.91, "advection_width": 1.24, "darcy": 6.27, "darcy_notch": 5.22, "navier_stokes": 9.12}


def stats(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def fmt(m, s=None, d=2, bold=False):
    t = f"{m:.{d}f}" + (f" ± {s:.{d}f}" if s is not None else "")
    return f"**{t}**" if bold else t


def gather(root):
    out = {}
    for b in BENCH:
        runs = [json.load(open(f)) for f in sorted(glob.glob(f"{root}/{b}_s*/metrics.json"))]
        if not runs:
            continue
        get = lambda f: stats([f(r) for r in runs])
        out[b] = {"n_seeds": len(runs), "operator": get(lambda r: 100 * r["operator_test"]),
                  "gpo_mll": get(lambda r: 100 * r["mll"]["gpo_test"]), "gpo": get(lambda r: 100 * r["gpo_test"])}
        for key in ("nll", "crps", "cov95", "cov90", "rho_field", "rho_point", "energy"):
            if key in runs[0]["uq_raw"]:
                out[b][f"raw_{key}"] = get(lambda r: r["uq_raw"][key])
            out[b][f"cal_{key}"] = get(lambda r: r["uq_calibrated"].get(key, np.nan))
            out[b][f"mll_cal_{key}"] = get(lambda r: r["mll"]["uq_calibrated"].get(key, np.nan))
        out[b]["tau"] = get(lambda r: r["tau"])
        out[b]["time_fit"] = get(lambda r: r["mll"]["time_fit"])
        dom = lambda r: r.get("gpo_test_domain", r.get("heldout", {}).get("gpo_test_domain"))
        if dom(runs[0]) is not None:
            out[b]["gpo_domain"] = get(lambda r: 100 * dom(r))
    return out


def tables(S, A):
    L = ["| Benchmark | GPO (paper) | **GPO (this release)** | Improvement | WNO (paper) | GP (paper) |",
         "|---|---|---|---|---|---|"]
    for b, r in S.items():
        m, s = r["gpo"]
        if b in PAPER:
            L.append(f"| {NAMES[b]} | {fmt(*PAPER[b])} | {fmt(m, s, bold=True)} | {PAPER[b][0] / m:.1f}× | "
                     f"{PAPER_WNO[b]:.2f} | {PAPER_GP[b]:.2f} |")
        else:
            L.append(f"| {NAMES[b]} | – | {fmt(m, s, bold=True)} | – | – | – |")
    acc = "\n".join(L)
    L = ["| Benchmark | NLL ↓ | CRPS ↓ | 95% coverage | 95% coverage (no recalibration) | ρ(σ, error) per field |",
         "|---|---|---|---|---|---|"]
    for b, r in S.items():
        L.append(f"| {NAMES[b]} | {fmt(*r['cal_nll'])} | {r['cal_crps'][0]:.2e} | {fmt(*r['cal_cov95'])} | "
                 f"{fmt(*r['raw_cov95'])} | {fmt(*r['raw_rho_field'])} |")
    uq = "\n".join(L)
    L = ["| Benchmark | input-embedding kernel, operator frozen | input-embedding kernel, trained jointly | + spatial kernel | + operator-feature kernel (GPO) |",
         "|---|---|---|---|---|"]
    for b in BENCH:
        f = lambda name: json.load(open(f"{A}/{b}_{name}/metrics.json")) if os.path.exists(f"{A}/{b}_{name}/metrics.json") else None
        row = [f("paper_kernel"), f("paper_kernel_joint"), f("no_linear")]
        vals = [fmt(100 * r.get("mll", r)["gpo_test"]) if r else "–" for r in row]
        full = S.get(b, {}).get("gpo", (np.nan,))[0]
        L.append(f"| {NAMES[b]} | " + " | ".join(vals) + f" | {fmt(full)} |")
    abl = "\n".join(L)
    return acc, uq, abl


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/gpo")
    p.add_argument("--ablation", default="runs/ablation")
    a = p.parse_args()
    S = gather(a.runs)
    os.makedirs("results", exist_ok=True)
    json.dump(S, open("results/summary.json", "w"), indent=1)
    acc, uq, abl = tables(S, a.ablation)
    open("results/tables.md", "w").write(f"## Accuracy\n\n{acc}\n\n## Uncertainty\n\n{uq}\n\n## Ablation\n\n{abl}\n")
    print(open("results/tables.md").read())
