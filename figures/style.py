"""Shared figure style: Inter, a quiet ink palette, perceptual colour maps, log-scale magma for sigma and |error|."""
import glob, os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "assets", "fonts", "Inter-*.ttf")):
    font_manager.fontManager.addfont(f)

INK, INK2, MUTED, GRID = "#1f2430", "#4b5563", "#6b7280", "#e5e7eb"
BLUE, ORANGE, BAND, INPUT, GOLD = "#3b5bdb", "#e8590c", "#3b5bdb", "#0f9d8a", "#f5b70a"
SERIES = [BLUE, ORANGE, INPUT, "#7048e8", GOLD, "#868e96", "#c2255c"]
CMAPS = {"navier_stokes": ("RdBu_r", "PuOr_r"), "darcy": ("viridis", "cividis"), "darcy_notch": ("viridis", "YlOrBr")}
S3 = 3 ** 0.5 / 2
NOTCH = [(0, 0), (0.49, 0), (0.49, 0.4), (0.51, 0.4), (0.51, 0), (1, 0), (0.5, S3)]
FPS = 24


def use():
    mpl.rcParams.update({
        "font.family": "Inter", "font.size": 11, "axes.edgecolor": "#c9cdd4", "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titleweight": "semibold",
        "axes.titlesize": 12.5, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False, "axes.grid": False, "grid.color": GRID,
        "grid.linewidth": 0.7, "axes.prop_cycle": mpl.cycler(color=SERIES), "savefig.dpi": 200, "savefig.bbox": "tight", "mathtext.fontset": "cm"})


def field_axes(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)


def grab(fig):
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return a[: a.shape[0] // 2 * 2, : a.shape[1] // 2 * 2].copy()


def write(frames, stem, gif_width=900, gif_step=2, still=-1):
    """frames -> stem.mp4 (H.264), stem.gif (downscaled) and stem_still.png (frame `still`)."""
    import imageio.v2 as imageio
    from PIL import Image
    imageio.mimwrite(stem + ".mp4", frames, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=2)
    ims = []
    for fr in frames[::gif_step]:
        im = Image.fromarray(fr)
        im = im.resize((gif_width, round(im.height * gif_width / im.width)), Image.LANCZOS)
        ims.append(im.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE))
    ims[0].save(stem + ".gif", save_all=True, append_images=ims[1:], duration=1000 * gif_step // FPS, loop=0, optimize=True)
    Image.fromarray(frames[still]).save(stem + "_still.png")


def schedule(n_items, hold=40, fade=10):
    """(i, j, w) per frame: item i blended towards item j with weight w; loops back to the first item."""
    out = []
    for i in range(n_items):
        out += [(i, i, 0.0)] * hold
        out += [(i, (i + 1) % n_items, (k + 1) / (fade + 1)) for k in range(fade)]
    return out
