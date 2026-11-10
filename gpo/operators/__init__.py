from .fno import FNO
from .wno import WNO
from .ewno import EWNO


def build_operator(cfg, grid, in_ch=1):
    """Builds an operator from a config dict with keys arch, width, level, layers, wavelet, modes, pad."""
    arch = cfg["arch"]
    if arch in ("wno", "wnof"):
        return WNO(grid, cfg["width"], cfg["level"], cfg["layers"], cfg["wavelet"], in_ch=in_ch, pad=cfg.get("pad", 0),
                   modes=cfg["modes"] if arch == "wnof" else None, fine=not cfg.get("no_fine", False))
    if arch == "ewno":
        return EWNO(grid, cfg["width"], cfg["level"], cfg["layers"], cfg.get("wavelet", "db4"), modes=cfg["modes"][0],
                    in_ch=in_ch, pad=cfg.get("pad", 0))
    if arch == "fno":
        return FNO(cfg["modes"], cfg["width"], cfg["layers"], in_ch=in_ch, pad=cfg.get("pad", 0))
    raise ValueError(f"unknown operator {arch}")


__all__ = ["FNO", "WNO", "EWNO", "build_operator"]
