"""Enhanced WNO: Fourier channel mixing on every wavelet level and band."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT, IDWT, DWT1D, IDWT1D

from .fno import grid_coords


def _cweights(cin, cout, *m):
    return nn.Parameter(torch.randn(cin, cout, *m, dtype=torch.cfloat) / (cin * cout) ** 0.5 / cin ** 0.5)


class WaveConv2dML(nn.Module):
    def __init__(self, width, level, size, wavelet="db4", modes=12, mode="symmetric"):
        super().__init__()
        self.dwt, self.idwt = DWT(J=level, mode=mode, wave=wavelet), IDWT(mode=mode, wave=wavelet)
        lo, hi = self.dwt(torch.zeros(1, 1, *size))
        shapes = [lo.shape[-2:]] + [h.shape[-2:] for h in hi]
        self.m = [(min(modes, s[0] // 2), min(modes, s[1] // 2 + 1)) for s in shapes]
        self.w = nn.ParameterList()
        for (m1, m2), nb in zip(self.m, [1] + [3] * level):
            for _ in range(2 * nb):
                self.w.append(_cweights(width, width, m1, m2))

    def _fconv(self, x, w1, w2, m):
        m1, m2 = m
        xf = torch.fft.rfft2(x)
        out = torch.zeros(x.shape[0], w1.shape[1], *xf.shape[-2:], dtype=torch.cfloat, device=x.device)
        out[..., :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[..., :m1, :m2], w1)
        out[..., -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[..., -m1:, :m2], w2)
        return torch.fft.irfft2(out, s=x.shape[-2:])

    def forward(self, x):
        lo, hi = self.dwt(x)
        it = iter(self.w)
        lo = self._fconv(lo, next(it), next(it), self.m[0])
        out_hi = [torch.stack([self._fconv(h[:, :, b], next(it), next(it), m) for b in range(3)], 2)
                  for h, m in zip(hi, self.m[1:])]
        return self.idwt((lo, out_hi))


class WaveConv1dML(nn.Module):
    def __init__(self, width, level, size, wavelet="db4", modes=16, mode="symmetric"):
        super().__init__()
        self.dwt, self.idwt = DWT1D(J=level, mode=mode, wave=wavelet), IDWT1D(mode=mode, wave=wavelet)
        lo, hi = self.dwt(torch.zeros(1, 1, size))
        self.m = [min(modes, s.shape[-1] // 2 + 1) for s in [lo] + hi]
        self.w = nn.ParameterList(_cweights(width, width, m) for m in self.m)

    def _fconv(self, x, w, m):
        xf = torch.fft.rfft(x)
        out = torch.zeros(x.shape[0], w.shape[1], xf.shape[-1], dtype=torch.cfloat, device=x.device)
        out[..., :m] = torch.einsum("bix,iox->box", xf[..., :m], w)
        return torch.fft.irfft(out, n=x.shape[-1])

    def forward(self, x):
        lo, hi = self.dwt(x)
        lo = self._fconv(lo, self.w[0], self.m[0])
        return self.idwt((lo, [self._fconv(h, w, m) for h, w, m in zip(hi, self.w[1:], self.m[1:])]))


class EWNO(nn.Module):
    def __init__(self, grid, width=32, level=3, layers=4, wavelet="db4", modes=12, in_ch=1, out_ch=1, pad=0, proj=128):
        super().__init__()
        self.dim, self.pad, self.width = len(grid), pad, width
        size = [g + pad for g in grid]
        conv = WaveConv1dML if self.dim == 1 else WaveConv2dML
        lin = nn.Conv1d if self.dim == 1 else nn.Conv2d
        self.lift = nn.Linear(in_ch + self.dim, width)
        self.wave = nn.ModuleList(conv(width, level, size[0] if self.dim == 1 else size, wavelet, modes) for _ in range(layers))
        self.skip = nn.ModuleList(lin(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, out_ch))

    def features(self, x):
        h = self.lift(torch.cat([x, grid_coords(x)], -1)).movedim(-1, 1)
        if self.pad:
            h = F.pad(h, [0, self.pad] * self.dim)
        for i, (w, s) in enumerate(zip(self.wave, self.skip)):
            h = w(h)[(...,) + tuple(slice(0, n) for n in h.shape[2:])] + s(h)
            if i < len(self.wave) - 1:
                h = F.gelu(h)
        if self.pad:
            h = h[(...,) + tuple(slice(0, n - self.pad) for n in h.shape[2:])]
        return h.movedim(1, -1)

    def forward(self, x):
        return self.proj(self.features(x))
