"""Figures for the theory checks and applications.

    python figures/plots.py --out assets/figures
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np
import torch
from figures.media import TITLES
from figures.style import MUTED, SERIES, save, use

load = lambda f: torch.load(f, weights_only=False)


def solvers(f, path):
    T = load(f)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.7), gridspec_kw={"wspace": 0.28})
    names = [("cg", "conjugate gradients", SERIES[6]), ("sdd", "SDD", SERIES[1]),
             ("sdd_backfit", "SDD, exact feature block", SERIES[0])]
    for k, label, col in names:
        h = T[k]
        t = np.array([r["time"] for r in h]) + 1e-2
        if k != "sdd_backfit":
            axes[0].plot(t, [max(r["gap"], 1e-30) for r in h], color=col, label=label, lw=1.8)
        axes[1].plot(t, [100 * r["rel_err"] for r in h], color=col, label=label, lw=1.8 if k != "cg" else 1.0)
    axes[0].set(xscale="log", yscale="log", xlabel="wall-clock (s)", title=r"$L_d(\alpha) - L_d^\star$")
    axes[1].set(xscale="log", yscale="log", xlabel="wall-clock (s)", title="Posterior mean, error vs exact (%)")
    for a in axes:
        a.grid(True, which="major")
    axes[1].legend(fontsize=9, loc="upper right")
    fig.suptitle(f"{TITLES.get(T['bench'], T['bench']).split(',')[0]},  N = {T['N']},  d = {T['d']}",
                 x=0.01, ha="left", fontsize=12.5, fontweight="semibold", y=1.02)
    save(fig, path)


def matheron_mc(files, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.1))
    for i, f in enumerate(files):
        T = load(f)
        mc = T["matheron_mc"]
        S = np.array([r["S"] for r in mc]); e = np.array([r["rel_fro"] for r in mc])
        ax.plot(S, e, "o-", color=SERIES[i], label=TITLES.get(T["bench"], T["bench"]), markersize=4)
    ref = e[0] * np.sqrt(S[0] / S)
    ax.plot(S, ref, ls="--", color=MUTED, lw=1, label="slope −1/2")
    ax.grid(True)
    ax.set(xscale="log", yscale="log", xlabel=r"draws $S$", ylabel="",
           title=r"$\|\hat C_S - C\|_F \,/\, \|C\|_F$")
    ax.legend(fontsize=7)
    save(fig, path)


def sensors(files, path):
    n = len(files)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 6.2), squeeze=False, gridspec_kw={"hspace": 0.35, "wspace": 0.25})
    styles = [("op_kriging", "operator + residual kriging", SERIES[1]), ("gpo_random", "GPO, random sensors", SERIES[0]),
              ("gpo_greedy", "GPO, uncertainty-guided sensors", SERIES[6])]
    for j, f in enumerate(files):
        T = load(f)
        m, S = np.array(T["m"]), T["summary"]
        ax = axes[0, j]
        ax.axhline(100 * S["operator"], color=MUTED, ls="--", lw=1.2, label="neural operator, no sensors")
        for k, label, col in styles:
            ax.plot(m, 100 * np.array(S[k]), "o-", color=col, label=label, markersize=4)
        ax.set(xscale="log", xlabel="number of sensors", title=TITLES.get(T["bench"], T["bench"])); ax.grid(True)
        if j == 0:
            ax.set_ylabel("relative L2 error (%)"); ax.legend(fontsize=8.5)
        ax = axes[1, j]
        ax.plot(m, S["cov95_random"], "o-", color=SERIES[0], markersize=4, label="random")
        ax.plot(m, S["cov95_greedy"], "o-", color=SERIES[6], markersize=4, label="uncertainty-guided")
        ax.axhline(0.95, color=MUTED, ls="--", lw=1)
        ax.set(xscale="log", ylim=(0.85, 1.0), xlabel="number of sensors"); ax.grid(True)
        if j == 0:
            ax.set_ylabel("95% interval coverage"); ax.legend(fontsize=8.5, loc="lower right")
    save(fig, path)


def functionals(files, path):
    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.4), squeeze=False)
    for j, f in enumerate(files):
        T = load(f)
        q = list(T["qoi"])
        x = np.arange(len(q))
        for k, (kind, col, label) in enumerate([("joint", SERIES[0], "joint Matheron draws"),
                                                 ("independent", SERIES[1], "same marginals, independent")]):
            v = [T["qoi"][n_][kind]["coverage"] for n_ in q]
            axes[0, j].bar(x + (k - 0.5) * 0.38, v, width=0.36, color=col, label=label)
        axes[0, j].axhline(T["level"], color=MUTED, ls="--", lw=1)
        axes[0, j].set_xticks(x, q, rotation=30, ha="right", fontsize=7.5)
        axes[0, j].set(ylim=(0, 1.02), title=TITLES.get(T["bench"], T["bench"]))
        axes[0, j].grid(axis="x", visible=False)
        if j == 0:
            axes[0, j].set_ylabel(f"coverage of {int(100 * T['level'])}% intervals")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, path)


def scaling(f, path):
    T = load(f)
    rows = T["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1))
    for name, col, label in [("trained", SERIES[0], "SDD, trained hyperparameters"), ("refit", SERIES[6], "SDD, re-estimated on unseen fields")]:
        r = [x for x in rows if x["hypers"] == name]
        if not r:
            continue
        N = [x["N"] for x in r]
        axes[0].plot(N, [100 * x["sdd_rl2"] for x in r], "o-", color=col, label=label, markersize=4)
        axes[1].plot(N, [x["sdd_mem"] for x in r], "o-", color=col, label="SDD" if name == "trained" else None, markersize=4)
        e = [x for x in r if x.get("exact_rl2") is not None]
        if e and name == "trained":
            axes[0].plot([x["N"] for x in e], [100 * x["exact_rl2"] for x in e], "s", color=SERIES[1], label="exact solve", markersize=6, mfc="none")
            axes[1].plot([x["N"] for x in e], [x["exact_mem"] for x in e], "s-", color=SERIES[1], label="exact solve", markersize=5)
    axes[0].axhline(100 * T["operator_test"], color=MUTED, ls="--", lw=1)
    axes[0].text(0.98, 100 * T["operator_test"], "operator alone", color=MUTED, fontsize=7, ha="right", va="bottom",
                 transform=axes[0].get_yaxis_transform())
    axes[0].set(xscale="log", xlabel="fields conditioned on, N", ylabel="test relative L2 (%)", title="Accuracy, operator fixed")
    axes[1].axhline(80, color=MUTED, ls=":", lw=1)
    axes[1].text(0.02, 80, "one A100, 80 GB", color=MUTED, fontsize=8, va="bottom", transform=axes[1].get_yaxis_transform())
    axes[1].set(xscale="log", xlabel="fields conditioned on, N", ylabel="peak GPU memory (GB)", title="Memory")
    Ns = sorted({x["N"] for x in rows})
    for a in axes:
        a.set_xticks(Ns, [f"{n // 1000}k" for n in Ns]); a.minorticks_off(); a.grid(True)
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, path)


def resolution(files, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"wspace": 0.25})
    for i, f in enumerate(files):
        T = load(f)
        rows = sorted(T["rows"], key=lambda r: r["points"])
        n0 = np.prod(T["grid"]) ** (1 / len(T["grid"]))
        x = [r["points"] ** (1 / len(T["grid"])) / n0 for r in rows]
        lab = TITLES.get(T["bench"], T["bench"]).split(",")[0] if T["bench"] != "advection" else "Wave advection, bump"
        if T["bench"] == "advection_width":
            lab = "Wave advection, width"
        axes[0].plot(x, [r["rl2"] for r in rows], "o-", color=SERIES[i], label=lab, markersize=4)
        axes[1].plot(x, [r["cov95"] for r in rows], "o-", color=SERIES[i], markersize=4)
    for a in axes:
        a.axvline(1, color=MUTED, ls="--", lw=1); a.set_xscale("log", base=2); a.grid(True)
        a.set_xlabel("input resolution / training resolution")
    axes[0].set(yscale="log", title="Test error (%)")
    axes[1].axhline(0.95, color=MUTED, ls=":", lw=1)
    axes[1].set(title="95% interval coverage", ylim=(0.5, 1.0))
    axes[1].legend(*axes[0].get_legend_handles_labels(), fontsize=8.5, loc="lower right")
    save(fig, path)


SHORT = {"navier_stokes": "Navier–Stokes", "darcy": "Darcy", "darcy_notch": "Darcy, notch", "burgers": "Burgers",
         "advection_width": "Advection, width", "advection": "Advection, bump"}


def ranking(runs, path):
    """Predicted against actual error of every test field: sigma ranks the hard cases."""
    import json
    from scipy.stats import spearmanr
    from gpo.data import load_split
    fig, axes = plt.subplots(1, len(runs), figsize=(3.1 * len(runs), 3.2), gridspec_kw={"wspace": 0.32})
    for a, run in zip(np.atleast_1d(axes), runs):
        M = json.load(open(f"{run}/metrics.json"))
        I = M["init"]
        yte = load_split(M["bench"], I["ntrain"], I["nval"], I["ntest"], tail_test=I["tail_test"])[2][1].flatten(1)
        P = load(f"{run}/pred.pt")
        sd = (P["sd"] * P["tau"]).double()
        rmse = (P["mean"].double() - yte.double()).pow(2).mean(1).sqrt().numpy()
        rsd = sd.pow(2).mean(1).sqrt().numpy()
        rho = spearmanr(rsd, rmse)[0]
        lo, hi = min(rsd.min(), rmse.min()) / 1.5, max(rsd.max(), rmse.max()) * 1.5
        a.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        a.scatter(rsd, rmse, s=10, color=SERIES[0], alpha=0.7, lw=0)
        a.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
        a.xaxis.set_minor_formatter(plt.NullFormatter()); a.yaxis.set_minor_formatter(plt.NullFormatter())
        a.set_title(f"{SHORT[M['bench']]}\nρ = {rho:.2f}", fontsize=10.5)
        a.set_xlabel("predicted σ", fontsize=9.5); a.tick_params(labelsize=8)
        a.grid(True)
    np.atleast_1d(axes)[0].set_ylabel("actual RMS error", fontsize=9.5)
    save(fig, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/figures")
    a = p.parse_args()
    use()
    os.makedirs(a.out, exist_ok=True)
    th = sorted(glob.glob("results/theory/*.pt"))
    for f in th:
        solvers(f, f"{a.out}/solvers_{os.path.basename(f)[:-3]}.png")
    if th:
        matheron_mc(th, f"{a.out}/matheron_convergence.png")
    s = [f"results/apps/sensors_{b}.pt" for b in ("navier_stokes", "darcy", "darcy_notch")]
    s = [f for f in s if os.path.exists(f)]
    if s:
        sensors(s, f"{a.out}/sensor_fusion.png")
    fn = sorted(glob.glob("results/apps/functionals_*.pt"))
    if fn:
        functionals(fn, f"{a.out}/functionals.png")
    rs = sorted(glob.glob("results/apps/resolution_*.pt"))
    if rs:
        resolution(rs, f"{a.out}/resolution.png")
    rk = [f"runs/gpo/{b}_s0" for b in ("darcy_notch", "burgers", "advection_width", "advection")]
    rk = [r for r in rk if os.path.exists(f"{r}/pred.pt")]
    if rk:
        ranking(rk, f"{a.out}/uncertainty_ranking.png")
    for f in glob.glob("results/apps/data_scaling_*.pt"):
        scaling(f, f"{a.out}/data_scaling.png")
    print("figures in", a.out)
