"""Matheron's rule on a one-dimensional toy problem: an animation and a four-panel figure.

    f | y  =  f  +  k(., X) K^-1 (y - f(X) - eps),     K = k(X, X) + s^2 I,   (f, eps) drawn from the prior.

Scenes: observations; a prior draw f; the residuals at the data, r = y - f(X) - eps; the correction k(., X) K^-1 r that
spreads the residuals with the kernel; the posterior draw f + correction; many draws against the exact posterior band.

    python figures/matheron_toy.py --out assets/media
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib.pyplot as plt
import numpy as np
from figures.style import BAND, BLUE, INK, MUTED, ORANGE, grab, save, use, write

ELL, NOISE = 0.11, 0.06


def matern52(a, b):
    r = np.sqrt(5) * np.abs(a[:, None] - b[None]) / ELL
    return (1 + r + r ** 2 / 3) * np.exp(-r)


def problem(seed=4):
    rng = np.random.default_rng(seed)
    truth = lambda t: np.sin(2 * np.pi * 1.15 * t + 0.3) + 0.45 * np.cos(9 * t) - 0.2
    X = np.array([0.05, 0.13, 0.22, 0.31, 0.43, 0.52, 0.86, 0.95])
    x = np.union1d(np.linspace(0, 1, 400), X)
    iX = np.searchsorted(x, X)
    y = truth(X) + NOISE * rng.standard_normal(len(X))
    L = np.linalg.cholesky(matern52(x, x) + 1e-9 * np.eye(len(x)))
    K = matern52(X, X) + NOISE ** 2 * np.eye(len(X))
    kx = matern52(x, X)
    mean = kx @ np.linalg.solve(K, y)
    var = 1 - np.einsum("ij,ji->i", kx, np.linalg.solve(K, kx.T))

    def draw(z, e):
        f = L @ z
        r = y - f[iX] - NOISE * e
        c = kx @ np.linalg.solve(K, r)
        return f, f[iX], r, c, f + c

    return x, X, y, mean, np.sqrt(np.clip(var, 0, None)), draw, rng


class Equation:
    """The rule as coloured pieces laid out left to right; pieces can be emphasised."""

    PIECES = [(r"$f \mid y$", INK), (r"$\;=\;$", MUTED), (r"$f$", BLUE), (r"$\;+\;$", MUTED),
              (r"$k(\cdot, X)\, K^{-1}$", ORANGE), (r"$\,($", MUTED), (r"$y - f(X) - \varepsilon$", ORANGE),
              (r"$)$", MUTED)]

    def __init__(self, fig, y=0.93, size=21):
        self.texts = [fig.text(0, y, s, color=c, fontsize=size, ha="left", va="center") for s, c in self.PIECES]
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        widths = [t.get_window_extent(r).width / fig.bbox.width for t in self.texts]
        x = 0.5 - sum(widths) / 2
        for t, w in zip(self.texts, widths):
            t.set_x(x); x += w

    def focus(self, idx):
        for i, t in enumerate(self.texts):
            t.set_alpha(1.0 if idx is None or i in idx else 0.22)


def animate(stem):
    x, X, y, mean, sd, draw, rng = problem()
    fig = plt.figure(figsize=(12.8, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.6, 1.15], hspace=0.12, left=0.06, right=0.985, top=0.83, bottom=0.08)
    a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for a in (a1, a2):
        a.grid(True); a.set_xlim(0, 1); a.tick_params(labelsize=9.5)
    a1.set_ylim(-2.9, 2.6); a2.set_ylim(-3.6, 3.6); a1.set_xticklabels([]); a2.set_xlabel(r"$x$")
    a1.set_ylabel(r"$f(x)$"); a2.set_ylabel("correction")
    a2.axhline(0, color=MUTED, lw=0.8)
    eq = Equation(fig)
    caption = fig.text(0.06, 0.865, "", ha="left", va="center", fontsize=13, color=INK, fontweight="semibold")
    data = a1.scatter(X, y, s=46, color=INK, edgecolor="white", lw=1.2, zorder=6, label=r"observations $y$")
    a1.legend(loc="lower left", fontsize=9.5)

    z0, e0 = rng.standard_normal(len(x)), rng.standard_normal(len(X))
    f, fX, r, c, post = draw(z0, e0)
    prior_line, = a1.plot([], [], color=BLUE, lw=2.0, zorder=4)
    post_line, = a1.plot([], [], color=INK, lw=2.2, zorder=5)
    stems = [a1.plot([], [], color=ORANGE, lw=2.2, solid_capstyle="round", zorder=5)[0] for _ in X]
    stems2 = [a2.plot([], [], color=ORANGE, lw=2.2, solid_capstyle="round")[0] for _ in X]
    tips2 = a2.scatter(X, r, s=26, color=ORANGE, zorder=5, alpha=0)
    corr_line, = a2.plot([], [], color=ORANGE, lw=2.0)
    cloud = []
    band = None
    frames = []

    def hold(n):
        frames.extend([grab(fig)] * n)

    def ease(k, n):
        return 0.5 - 0.5 * np.cos(np.pi * (k + 1) / n)

    eq.focus(None); caption.set_text("")
    data.set_alpha(0); hold(30)
    data.set_alpha(1); eq.focus({0}); caption.set_text(r"Observations $y$"); hold(30)

    eq.focus({2}); caption.set_text(r"1   Prior draw  $f \sim \mathcal{GP}(0, k)$")
    for k in range(24):
        prior_line.set_data(x, f); prior_line.set_alpha(ease(k, 24)); frames.append(grab(fig))
    hold(24)

    eq.focus({6}); caption.set_text(r"2   Residuals  $r = y - f(X) - \varepsilon$")
    for k in range(24):
        t = ease(k, 24)
        for s, xi, a, b in zip(stems, X, fX, fX + r):
            s.set_data([xi, xi], [a, a + t * (b - a)])
        frames.append(grab(fig))
    hold(24)

    eq.focus({4, 5, 6, 7}); caption.set_text(r"3   Correction  $k(\cdot, X)\, K^{-1} r$")
    for k in range(20):
        t = ease(k, 20)
        for s, xi, ri in zip(stems2, X, r):
            s.set_data([xi, xi], [0, t * ri])
        frames.append(grab(fig))
    tips2.set_alpha(1)
    for k in range(30):
        n = int(len(x) * ease(k, 30))
        corr_line.set_data(x[:n], c[:n]); frames.append(grab(fig))
    hold(24)

    eq.focus({0, 1, 2, 3, 4, 5, 6, 7}); caption.set_text(r"4   Posterior draw  $f + k(\cdot, X)\, K^{-1} r$")
    for k in range(36):
        t = ease(k, 36)
        prior_line.set_data(x, f + t * c); prior_line.set_color(BLUE if t < 0.5 else INK)
        for s, xi, a, b in zip(stems, X, fX + t * r, fX + r):
            s.set_data([xi, xi], [a, b])
        frames.append(grab(fig))
    for s in stems:
        s.set_data([], [])
    prior_line.set_data([], []); post_line.set_data(x, post); hold(30)

    caption.set_text(r"5   Repeat with new $(f, \varepsilon)$")
    for i in range(40):
        fi, fXi, ri, ci, pi = draw(rng.standard_normal(len(x)), rng.standard_normal(len(X)))
        cloud.append(a1.plot(x, pi, color=INK, lw=0.7, alpha=0.28, zorder=3)[0])
        corr_line.set_data(x, ci); tips2.set_offsets(np.c_[X, ri])
        for s, xi, rr in zip(stems2, X, ri):
            s.set_data([xi, xi], [0, rr])
        frames.extend([grab(fig)] * (3 if i < 10 else 1))
    post_line.set_data([], [])
    for h in stems2 + [corr_line]:
        h.set_alpha(0.25)
    tips2.set_alpha(0.25)
    caption.set_text("40 draws  vs  exact posterior")
    for k in range(24):
        t = ease(k, 24)
        if band is not None:
            band.remove()
        band = a1.fill_between(x, mean - 1.96 * sd, mean + 1.96 * sd, color=BAND, alpha=0.18 * t, lw=0, zorder=2)
        frames.append(grab(fig))
    a1.plot(x, mean, color=BLUE, lw=1.8, ls=(0, (4, 2)), zorder=6, label="exact mean")
    a1.legend(loc="lower left", fontsize=9.5)
    hold(72)
    plt.close(fig)
    write(frames, stem, gif_width=960)


def panels(path):
    """Four-panel summary of the same steps."""
    x, X, y, mean, sd, draw, rng = problem()
    z0, e0 = rng.standard_normal(len(x)), rng.standard_normal(len(X))
    f, fX, r, c, post = draw(z0, e0)
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9), sharey=True, gridspec_kw={"wspace": 0.1})
    titles = [r"Prior draw  $f$", r"Residuals  $y - f(X) - \varepsilon$", r"Correction  $k(\cdot, X)\, K^{-1} r$",
              r"Posterior draws"]
    for a, t in zip(ax, titles):
        a.grid(True); a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_ylim(-2.9, 2.6); a.set_title(t, loc="left", fontsize=13); a.set_xlabel(r"$x$")
    ax[0].plot(x, f, color=BLUE, lw=2); ax[0].scatter(X, y, s=40, color=INK, edgecolor="white", zorder=5)
    ax[1].plot(x, f, color=BLUE, lw=1.4, alpha=0.5); ax[1].scatter(X, y, s=40, color=INK, edgecolor="white", zorder=5)
    for xi, a, b in zip(X, fX, fX + r):
        ax[1].plot([xi, xi], [a, b], color=ORANGE, lw=2.4, solid_capstyle="round")
    ax[2].axhline(0, color=MUTED, lw=0.8)
    for xi, ri in zip(X, r):
        ax[2].plot([xi, xi], [0, ri], color=ORANGE, lw=2.4, solid_capstyle="round")
    ax[2].plot(x, c, color=ORANGE, lw=2)
    ax[3].fill_between(x, mean - 1.96 * sd, mean + 1.96 * sd, color=BAND, alpha=0.18, lw=0, label="exact 95%")
    for _ in range(30):
        ax[3].plot(x, draw(rng.standard_normal(len(x)), rng.standard_normal(len(X)))[4], color=INK, lw=0.6, alpha=0.3)
    ax[3].plot(x, post, color=INK, lw=2, label="draw of panel 1")
    ax[3].scatter(X, y, s=40, color=INK, edgecolor="white", zorder=5); ax[3].legend(loc="lower left", fontsize=9)
    fig.suptitle(r"$f \mid y \;=\; f + k(\cdot, X)\, K^{-1} (y - f(X) - \varepsilon)$", x=0.5, fontsize=19, y=1.06)
    save(fig, path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets/media")
    a = p.parse_args()
    use()
    os.makedirs(a.out, exist_ok=True)
    panels(f"{a.out}/matheron_toy_panels.png")
    animate(f"{a.out}/matheron_toy")
    print("wrote", a.out)
