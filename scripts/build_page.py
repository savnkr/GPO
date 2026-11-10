"""Writes docs/index.html from scripts/page_template.html and results/summary.json and copies figures and videos into docs/static.

    python scripts/collect.py && python scripts/build_page.py
"""
import glob, json, os, shutil

TEMPLATE, PAGE, STATIC = "scripts/page_template.html", "docs/index.html", "docs/static"
NAMES = {"navier_stokes": "Navier–Stokes", "darcy": "Darcy flow", "darcy_notch": "Darcy, notched domain",
         "burgers": "Burgers", "advection_width": "Wave advection, varying width",
         "advection": "Wave advection, wave and bump"}
ORDER = ["navier_stokes", "darcy", "darcy_notch", "burgers", "advection_width", "advection"]
PAPER = {"navier_stokes": ("2.21 ± 0.44", "12.23", "9.12"), "darcy": ("3.98 ± 0.12", "2.82", "6.27"),
         "darcy_notch": ("2.18 ± 0.19", "2.21", "5.22"), "burgers": ("2.89 ± 0.11", "4.09", "4.91"),
         "advection_width": ("0.63 ± 0.08", "1.01", "1.24")}
CAPTIONS = {
    "navier_stokes": "Forcing to vorticity at T = 10, 64 × 64.",
    "darcy": "Permeability to pressure, 85 × 85, standard split.",
    "darcy_notch": "Boundary condition to pressure on a triangle with a notch, 51 × 51.",
    "burgers": "Initial condition to the solution at T = 1, 1024 points.",
    "advection_width": "Square wave of random width, height and position, advected to T = 0.5, 200 points.",
    "advection": "Square wave plus a bump, advected to T = 0.5, 200 points.",
}

ONE_D = ("burgers", "advection_width", "advection")
LEGEND = {"2d": "If the uncertainty is calibrated, the actual error looks like the sampled one and leaves the 95% "
                "interval at about 5% of points.",
          "1d": "Shaded: 95% interval of the posterior."}


def copy_media():
    os.makedirs(f"{STATIC}/videos", exist_ok=True)
    os.makedirs(f"{STATIC}/images", exist_ok=True)
    for f in glob.glob("assets/media/*.mp4"):
        shutil.copy(f, f"{STATIC}/videos/")
    for f in glob.glob("assets/media/*.png") + glob.glob("assets/figures/*.png"):
        shutil.copy(f, f"{STATIC}/images/")


def fill(html, key, content):
    return html.replace(f"<!--{key}-->", content)


def sensor_stat():
    import torch
    T = torch.load("results/apps/sensors_navier_stokes.pt", weights_only=False)
    S = T["summary"]
    drop = 100 * (1 - S["gpo_greedy"][-1] / S["gpo"])
    return f"{drop:.0f}%", f"error reduction on Navier–Stokes with {T['m'][-1]} sensors"


def main():
    copy_media()
    S = json.load(open("results/summary.json")) if os.path.exists("results/summary.json") else {}
    html = open(TEMPLATE).read()
    acc, uq = [], []
    for b in ORDER:
        if b not in S:
            continue
        m, s = S[b]["gpo"]
        p = PAPER.get(b, ("–", "–", "–"))
        acc.append(f'<tr><td>{NAMES[b]}</td><td class="ours">{m:.2f} ± {s:.2f}</td><td class="pub">{p[0]}</td>'
                   f'<td class="pub">{p[1]}</td><td class="pub">{p[2]}</td></tr>')
        uq.append(f"<tr><td>{NAMES[b]}</td><td>{S[b]['cal_nll'][0]:.2f}</td><td>{S[b]['cal_crps'][0]:.2e}</td>"
                  f"<td>{S[b]['cal_cov95'][0]:.3f}</td><td>{S[b]['raw_rho_field'][0]:.2f}</td></tr>")
    stats = []
    if "navier_stokes" in S and "darcy_notch" in S:
        r = [float(PAPER[b][0].split()[0]) / S[b]["gpo"][0] for b in PAPER if b in S]
        stats = [(f"{min(r):.0f}–{max(r):.0f}×", "lower error than reported in the paper"),
                 (f"{S['navier_stokes']['gpo'][0]:.2f}%", "Navier–Stokes test error (paper: 2.21%)"),
                 ("0", "approximations in the posterior and its draws"),
                 sensor_stat()]
    stats_html = "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in stats)
    panes = []
    for i, b in enumerate(ORDER):
        if not os.path.exists(f"{STATIC}/videos/gallery_{b}.mp4"):
            continue
        panes.append(f'<div class="tab-pane{" active" if i == 0 else ""}" id="g-{b}"><div class="media-frame">'
                     f'<video class="media" {"autoplay " if i == 0 else ""}muted loop playsinline preload="metadata" '
                     f'poster="static/images/gallery_{b}_still.png"><source src="static/videos/gallery_{b}.mp4" '
                     f'type="video/mp4"></video></div><p class="caption"><b>{NAMES[b]}.</b> {CAPTIONS[b]} {LEGEND["1d" if b in ONE_D else "2d"]}</p></div>')
    html = fill(html, "ACC_ROWS", "\n          ".join(acc))
    html = fill(html, "UQ_ROWS", "\n          ".join(uq))
    html = fill(html, "STATS", stats_html)
    html = fill(html, "GALLERY_PANES", "\n    ".join(panes))
    open(PAGE, "w").write(html)
    print("wrote", PAGE)


if __name__ == "__main__":
    main()
