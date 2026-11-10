"""Galleries, Matheron-rule morphs and the sensor-fusion animation for a trained GPO run.

    python figures/media.py --run runs/gpo/navier_stokes_s0 --out assets/media
    python figures/media.py --sensors results/apps/sensors_navier_stokes.pt --out assets/media
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.ndimage import distance_transform_edt
from experiments.common import Run
from figures.style import BAND, BLUE, CMAPS, INK, INPUT, MUTED, NOTCH, ORANGE, S3, field_axes, grab, schedule, use, write

TITLES = {"burgers": "Burgers' equation", "advection_width": "Wave advection, varying width",
          "advection": "Wave advection, wave and bump", "darcy": "Darcy flow", "darcy_notch": "Darcy flow in a notched domain",
          "navier_stokes": "Navier–Stokes, vorticity from forcing"}


def picks(err, n):
    """n test cases spread over the error distribution (10th to 90th percentile)."""
    order = np.argsort(err)
    return [int(order[int(q * (len(err) - 1))]) for q in np.linspace(0.1, 0.9, n)]


def load(run, n=6, n_draws=8, seed=0):
    R = Run(run)
    m = R.model
    mu_all, _ = m.predict(R.Xte, var=False)
    err = ((R.decode(mu_all) - R.yte).norm(dim=1) / R.yte.norm(dim=1)).cpu().numpy()
    idx = picks(err, n)
    X = R.Xte[idx]
    mu, var = m.predict(X)
    g = torch.Generator(device=X.device).manual_seed(seed)
    post, prior = m.sample(X, n_samples=n_draws, generator=g, return_prior=True)
    npy = lambda t: t.float().cpu().numpy()
    dom = None
    if R.bench == "darcy_notch":
        from gpo.data import notch_mask
        dom = notch_mask().numpy()
    sd = R.sd(var, X)
    mu_v, var_v = m.predict(R.Xva)
    tau_f = ((R.decode(mu_v) - R.yva) ** 2 / (var_v * R.s_out ** 2)).mean().sqrt()
    sd_f = var.sqrt() * R.s_out * tau_f
    err_draw = tau_f * (post[0] - mu) * R.s_out
    return dict(bench=R.bench, grid=R.grid, dom=dom, x=R.xte_raw[idx].squeeze(-1).numpy(), y=npy(R.yte[idx]),
                mu=npy(R.decode(mu)), sd=npy(sd), sdf=npy(sd_f), err_draw=npy(err_draw),
                post=npy(R.decode(post)), prior=npy(R.decode(prior)), err=err[idx])


def fill_outside(f, dom):
    if dom is None:
        return f
    _, near = distance_transform_edt(~dom, return_indices=True)
    return f[near[0], near[1]]


def as_field(D, v):
    return fill_outside(np.asarray(v, dtype=float).reshape(D["grid"]), D["dom"])


def draw_field(ax, img, cmap, norm, dom):
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    im = ax.imshow(img, origin="lower", extent=(0, 1, 0, 1), cmap=cm, norm=norm, interpolation="bicubic")
    if dom is not None:
        poly = Polygon(NOTCH, closed=True, fc="none", ec="#9ca3af", lw=0.8)
        ax.add_patch(poly); im.set_clip_path(poly); ax.set_ylim(0, S3 + 0.02)
    field_axes(ax)
    return im


# ----------------------------------------------------------------------------------------------------------------------
def gallery_2d(D, stem):
    """Input, truth, mean; the actual error next to a posterior sample of the error (same scale); the points where the
    truth leaves the 95% interval; and both errors with the interval along one line of the field. The simulated truth
    is noise-free, so the comparison uses the posterior of the field, scaled once on the validation fields."""
    bench, dom = D["bench"], D["dom"]
    cmap, cmap_in = CMAPS[bench]
    n = len(D["y"])
    H, W = D["grid"]
    row = int(0.22 * (H - 1)) if dom is not None else H // 2
    yline = row / (H - 1)
    F = {"inp": [as_field(D, D["x"][i]) for i in range(n)], "y": [as_field(D, D["y"][i]) for i in range(n)],
         "mu": [as_field(D, D["mu"][i]) for i in range(n)],
         "err": [as_field(D, D["y"][i] - D["mu"][i]) for i in range(n)],
         "dev": [as_field(D, D["err_draw"][i]) for i in range(n)],
         "out": [as_field(D, (np.abs(D["y"][i] - D["mu"][i]) > 1.96 * D["sdf"][i]).astype(float)) for i in range(n)],
         "sd": [as_field(D, D["sdf"][i]) for i in range(n)]}
    inside = (lambda a: a[:, dom]) if dom is not None else (lambda a: a.reshape(len(a), -1))
    pct = lambda k, q: np.percentile(inside(np.stack(F[k])), q)
    signed = cmap == "RdBu_r"
    lo, hi = (-pct("y", 99.5), pct("y", 99.5)) if signed else (pct("y", 0.5), pct("y", 99.5))
    lo_i, hi_i = pct("inp", 0.5), pct("inp", 99.5)
    el = max(pct("err", 99.5), -pct("err", 0.5), pct("dev", 99.5), -pct("dev", 0.5))
    field, errn = Normalize(lo, hi), Normalize(-el, el)
    out_cm = ListedColormap(["#eef0f5", ORANGE])
    panels = [("inp", "Input", cmap_in, Normalize(lo_i, hi_i)), ("y", "Truth", cmap, field),
              ("mu", "Posterior mean", cmap, field), ("err", "Actual error", "PuOr_r", errn),
              ("dev", "Sampled error", "PuOr_r", errn), ("out", "Outside 95% interval", out_cm, Normalize(0, 1))]
    subtitles = ["", "", "", "truth − mean", "draw − mean", ""]
    fig = plt.figure(figsize=(19.5, 5.6))
    gs = fig.add_gridspec(2, 9, width_ratios=[1, 1, 1, 0.3, 0.1, 1, 1, 0.3, 1], height_ratios=[1, 0.42],
                          wspace=0.06, hspace=0.32, left=0.01, right=0.99, top=0.88, bottom=0.09)
    cols = (0, 1, 2, 5, 6, 8)
    ax = [fig.add_subplot(gs[0, c]) for c in cols]
    ims = [draw_field(a, F[k][0], cm, nm, dom) for a, (k, t, cm, nm) in zip(ax, panels)]
    titles = []
    for a, (_, t, _, _), st in zip(ax, panels, subtitles):
        titles.append(a.set_title(t + (f"\n{st}" if st else "\n "), linespacing=1.5))
    for k in (3, 4):
        ax[k].axhline(yline, color=INK, lw=0.9, ls=(0, (4, 3)))
    fig.canvas.draw()
    for k in (2, 4):
        pos = ax[k].get_position()
        cb = fig.colorbar(ims[k], cax=fig.add_axes([pos.x1 + 0.006, pos.y0 + 0.1 * pos.height, 0.006, 0.8 * pos.height]))
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8.5)
    ls = fig.add_subplot(gs[1, 1:])
    xs = np.linspace(0, 1, W)
    ls.set_xlim(0, 1); ls.set_ylim(-1.15 * el, 1.15 * el); ls.grid(True); ls.tick_params(labelsize=9)
    ls.axhline(0, color=MUTED, lw=0.6)
    ls.set_xlabel("position along the dashed line", fontsize=10)
    ls.legend([Patch(color=BAND, alpha=0.2), Line2D([], [], color=INK, lw=1.6), Line2D([], [], color=BLUE, lw=1.0)],
              ["95% interval", "actual error", "sampled error"], loc="center left", bbox_to_anchor=(-0.13, 0.5),
              fontsize=10, frameon=False)
    ls.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.0e") if el < 1e-2 else mpl.ticker.ScalarFormatter())
    line = lambda k, i: F[k][i][row]
    dyn = []

    def paint(i, alpha):
        sd = line("sd", i)
        return [ls.fill_between(xs, -1.96 * sd, 1.96 * sd, color=BAND, alpha=0.2 * alpha, lw=0),
                ls.plot(xs, line("dev", i), color=BLUE, lw=1.0, alpha=0.8 * alpha)[0],
                ls.plot(xs, line("err", i), color=INK, lw=1.6, alpha=alpha)[0]]

    frac = [100 * float(np.mean(inside(F["out"][i][None]))) for i in range(n)]
    frames = []
    for i, j, w in schedule(n):
        for im, (k, *_) in zip(ims, panels):
            im.set_data((1 - w) * F[k][i] + w * F[k][j])
        titles[5].set_text(f"Outside 95% interval\n{(1 - w) * frac[i] + w * frac[j]:.1f}% of points")
        for h in dyn:
            h.remove()
        dyn = (paint(i, 1 - w) if w < 1 else []) + (paint(j, w) if w > 0 else [])
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, stem, still=0)


def gallery_1d(D, stem):
    xs = np.linspace(0, 1, D["y"].shape[1])
    data = [(D["x"][i], D["y"][i], D["mu"][i], D["sdf"][i], D["post"][:, i]) for i in range(len(D["y"]))]
    sets = [data[0::2], data[1::2]]
    ncol = len(sets[0])
    fig = plt.figure(figsize=(14, 5.4))
    gs = fig.add_gridspec(3, ncol, height_ratios=[0.75, 2.2, 1.15], hspace=0.14, wspace=0.08, left=0.065, right=0.995,
                          top=0.88, bottom=0.1)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(ncol)] for r in range(3)])
    rng = lambda vs, pad: (min(v.min() for v in vs) - pad * np.ptp(np.concatenate(vs)),
                           max(v.max() for v in vs) + pad * np.ptp(np.concatenate(vs)))
    lim_in = rng([d[0] for d in data], 0.12); lim_out = rng([d[1] for d in data] + [d[2] for d in data], 0.08)
    em = 1.15 * max(max(np.percentile(np.abs(d[2] - d[1]), 99.5), np.percentile(1.96 * d[3], 99.5)) for d in data)
    art = []
    for c in range(ncol):
        a0, a1, a2 = ax[:, c]
        for a in (a0, a1, a2):
            a.grid(True); a.set_xlim(0, 1); a.tick_params(labelsize=9.5)
            if c:
                a.set_yticklabels([])
        a0.set_ylim(*lim_in); a1.set_ylim(*lim_out); a2.set_ylim(-em, em)
        a0.set_xticklabels([]); a1.set_xticklabels([]); a2.set_xlabel("x")
        a0.spines["left"].set_visible(False); a0.set_yticks([])
        a2.axhline(0, color=MUTED, lw=0.7)
        art.append([a0, a1, a2])
    ax[0, 0].set_ylabel("input", labelpad=8); ax[1, 0].set_ylabel("output"); ax[2, 0].set_ylabel("error")
    fig.legend([Line2D([], [], color=INPUT, lw=1.5), Line2D([], [], color=INK, lw=2), Line2D([], [], color=BLUE, lw=1.6, ls=(0, (4, 2))),
                Patch(color=BAND, alpha=0.25), Line2D([], [], color=ORANGE, lw=1.2)],
               ["input", "truth", "posterior mean", "95% interval", "mean − truth"], loc="upper center", ncol=5,
               fontsize=10.5, bbox_to_anchor=(0.52, 1.0))

    def paint(c, d, alpha):
        a0, a1, a2 = art[c]
        u, y, mu, sd, dr = d
        out = [a0.fill_between(xs, lim_in[0], u, color=INPUT, alpha=0.16 * alpha, lw=0), a0.plot(xs, u, color=INPUT, lw=1.5, alpha=alpha)[0],
               a1.fill_between(xs, mu - 1.96 * sd, mu + 1.96 * sd, color=BAND, alpha=0.2 * alpha, lw=0),
               a2.fill_between(xs, -1.96 * sd, 1.96 * sd, color=BAND, alpha=0.2 * alpha, lw=0)]
        out += [a1.plot(xs, y, color=INK, lw=2.0, alpha=alpha)[0], a1.plot(xs, mu, color=BLUE, lw=1.6, ls=(0, (4, 2)), alpha=alpha)[0]]
        out.append(a2.plot(xs, mu - y, color=ORANGE, lw=1.3, alpha=alpha)[0])
        return out

    frames, dyn = [], []
    for i, j, w in schedule(len(sets), hold=72, fade=14):
        for h in dyn:
            h.remove()
        dyn = []
        for c in range(ncol):
            if w < 1:
                dyn += paint(c, sets[i][c], 1 - w)
            if w > 0:
                dyn += paint(c, sets[j][c], w)
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, stem, still=0)


# ----------------------------------------------------------------------------------------------------------------------
def morph_times(hold=24, steps=36):
    return [0.0] * hold + list(0.5 - 0.5 * np.cos(np.linspace(0, np.pi, steps))) + [1.0] * (2 * hold)


def morph_label(t):
    if t == 0:
        return r"$f \sim \mathcal{GP}(0, k)$"
    if t == 1:
        return r"$f \mid y$"
    return rf"$f + {t:.2f}\; k(\cdot, X)\, K^{{-1}} (y - f(X) - \varepsilon)$"


def matheron_2d(D, stem, n=4):
    """Prior draws of one test field morph into posterior draws: f_t = f + t (f | y - f)."""
    bench, dom = D["bench"], D["dom"]
    cmap = CMAPS[bench][0]
    y = as_field(D, D["y"][0])
    pri = [as_field(D, D["prior"][s, 0]) for s in range(n)]
    post = [as_field(D, D["post"][s, 0]) for s in range(n)]
    allv = np.stack([y] + pri + post)
    if cmap == "RdBu_r":
        lim = np.percentile(np.abs(allv), 99.5); norm = Normalize(-lim, lim)
    else:
        norm = Normalize(*np.percentile(allv, [0.5, 99.5]))
    fig, ax = plt.subplots(1, n + 1, figsize=(3.1 * (n + 1), 3.75), gridspec_kw={"wspace": 0.06})
    draw_field(ax[0], y, cmap, norm, dom); ax[0].set_title("Truth")
    ims = [draw_field(ax[s + 1], pri[s], cmap, norm, dom) for s in range(n)]
    for s in range(n):
        ax[s + 1].set_title(f"Draw {s + 1}")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.03)
    label = fig.text(0.01, 0.925, "", fontsize=19, color=INK, va="center")
    frames = []
    for t in morph_times():
        for s in range(n):
            ims[s].set_data((1 - t) * pri[s] + t * post[s])
        label.set_text(morph_label(t))
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, stem)


def matheron_1d(D, stem, n=8):
    xs = np.linspace(0, 1, D["y"].shape[1])
    y, pri, post = D["y"][0], D["prior"][:n, 0], D["post"][:n, 0]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    lo = min(y.min(), pri.min(), post.min()); hi = max(y.max(), pri.max(), post.max())
    ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)); ax.set_xlim(0, 1); ax.grid(True); ax.set_xlabel("x")
    lines = [ax.plot(xs, pri[s], color=BLUE, lw=1.0, alpha=0.75)[0] for s in range(n)]
    ax.plot(xs, y, color=INK, lw=1.8, ls=(0, (4, 2)), label="truth"); ax.legend(loc="upper right", fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.15)
    label = fig.text(0.07, 0.915, "", fontsize=18, color=INK, va="center")
    frames = []
    for t in morph_times():
        for s in range(n):
            lines[s].set_ydata((1 - t) * pri[s] + t * post[s])
        label.set_text(morph_label(t))
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, stem)


# ----------------------------------------------------------------------------------------------------------------------
def sensors_animation(pt, stem):
    """Frames saved by experiments/sensor_fusion.py: sensors added one at a time where the variance is largest."""
    T = torch.load(pt, weights_only=False)
    F, bench = T["frames"], T["bench"]
    grid = tuple(F["grid"])
    if len(grid) != 2:
        return
    dom = None
    if bench == "darcy_notch":
        from gpo.data import notch_mask
        dom = notch_mask().numpy()
    D = {"grid": grid, "dom": dom}
    cmap = CMAPS[bench][0]
    y = as_field(D, F["truth"])
    norm = Normalize(-np.abs(y).max(), np.abs(y).max()) if cmap == "RdBu_r" else Normalize(y.min(), y.max())
    emax = np.percentile(np.abs(as_field(D, F["mean"][0]) - y), 99.5)
    smax = np.percentile(as_field(D, F["sd"][0]), 99.5)
    J = np.asarray(F["J"])
    rows, cols = np.unravel_index(J, grid)
    px, py = (cols + 0.5) / grid[1], (rows + 0.5) / grid[0]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.75), gridspec_kw={"wspace": 0.06})
    draw_field(ax[0], y, cmap, norm, dom); ax[0].set_title("Truth")
    ims = [draw_field(ax[1], y, cmap, norm, dom), draw_field(ax[2], y, cmap, norm, dom),
           draw_field(ax[3], y, "magma", Normalize(0, smax), dom), draw_field(ax[4], y, "magma", Normalize(0, emax), dom)]
    dots = [a.plot([], [], "o", ms=4, mfc="white", mec=INK, mew=0.9)[0] for a in ax[1:]]
    titles = [ax[1].set_title(""), ax[2].set_title("Posterior draw"), ax[3].set_title("σ"), ax[4].set_title("")]
    fig.subplots_adjust(left=0.01, right=0.99, top=0.8, bottom=0.03)
    frames = []
    ks = list(range(len(F["mean"])))
    for k in [0] * 18 + [kk for kk in ks for _ in range(3)] + [ks[-1]] * 36:
        mu, sd, dr = as_field(D, F["mean"][k]), as_field(D, F["sd"][k]), as_field(D, F["draws"][k][0])
        rel = np.linalg.norm(mu - y) / np.linalg.norm(y)
        for im, img in zip(ims, (mu, dr, sd, np.abs(mu - y))):
            im.set_data(img)
        for d in dots:
            d.set_data(px[:k], py[:k])
        titles[0].set_text(f"Mean, {k} sensors"); titles[3].set_text(f"|Error|   {100 * rel:.2f}%")
        frames.append(grab(fig))
    plt.close(fig)
    write(frames, stem)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="trained run: galleries and Matheron morphs")
    p.add_argument("--out", default="assets/media")
    p.add_argument("--sensors", default=None, help="render only the sensor animation from this results file")
    p.add_argument("--gallery-only", action="store_true")
    a = p.parse_args()
    if not (a.run or a.sensors):
        p.error("give --run or --sensors")
    use()
    os.makedirs(a.out, exist_ok=True)
    if a.sensors:
        T = torch.load(a.sensors, weights_only=False)
        sensors_animation(a.sensors, f"{a.out}/sensors_{T['bench']}")
        sys.exit()
    D = load(a.run)
    b = D["bench"]
    if len(D["grid"]) == 1:
        gallery_1d(D, f"{a.out}/gallery_{b}")
        if not a.gallery_only:
            matheron_1d(D, f"{a.out}/matheron_{b}")
    else:
        gallery_2d(D, f"{a.out}/gallery_{b}")
        if not a.gallery_only:
            matheron_2d(D, f"{a.out}/matheron_{b}")
    print("wrote", a.out, b)
