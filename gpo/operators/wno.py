"""Wavelet neural operator (Tripura & Chakraborty, 2023) on 1D/2D grids, with an optional spectral path (WNO+F)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward, DWT1DInverse, DWTForward, DWTInverse

from .fno import SpectralConv, grid_coords


class WaveConv(nn.Module):
    """Location-wise channel mixing on the coarsest approximation and details; a shared channel map on finer details."""

    def __init__(self, width, grid, level, wavelet="db4", fine=True):
        super().__init__()
        self.dim, self.fine = len(grid), fine
        if self.dim == 1:
            self.dwt, self.idwt = DWT1DForward(J=level, wave=wavelet, mode="symmetric"), DWT1DInverse(wave=wavelet, mode="symmetric")
        else:
            self.dwt, self.idwt = DWTForward(J=level, wave=wavelet, mode="symmetric"), DWTInverse(wave=wavelet, mode="symmetric")
        with torch.no_grad():
            coarse = self.dwt(torch.zeros(1, 1, *grid))[0].shape[2:]
        n_orient = 1 if self.dim == 1 else 3
        s = 1 / width
        self.w_approx = nn.Parameter(s * torch.randn(width, width, *coarse))
        self.w_detail = nn.Parameter(s * torch.randn(n_orient, width, width, *coarse))
        if fine:
            self.w_fine = nn.Parameter(s * torch.randn(level - 1, n_orient, width, width))
        self.eq = "bix,iox->box" if self.dim == 1 else "bixy,ioxy->boxy"

    def forward(self, x):
        yl, yh = self.dwt(x)
        out_l = torch.einsum(self.eq, yl, self.w_approx)
        out_h = []
        for j, c in enumerate(yh):
            if j == len(yh) - 1:
                if self.dim == 1:
                    o = torch.einsum(self.eq, c, self.w_detail[0])
                else:
                    o = torch.stack([torch.einsum(self.eq, c[:, :, k], self.w_detail[k]) for k in range(3)], 2)
            elif self.fine:
                o = (torch.einsum("bix,io->box", c, self.w_fine[j, 0]) if self.dim == 1
                     else torch.einsum("bik...,kio->bok...", c, self.w_fine[j]))
            else:
                o = torch.zeros_like(c)
            out_h.append(o)
        y = self.idwt((out_l, out_h))
        return y[(...,) + tuple(slice(0, n) for n in x.shape[2:])]


class WNO(nn.Module):
    """(b, *grid, in_ch) -> (b, *grid, out_ch). `features` returns the last hidden layer, (b, *grid, width)."""

    def __init__(self, grid, width=64, level=3, layers=4, wavelet="db4", in_ch=1, out_ch=1, proj=128, pad=0,
                 modes=None, fine=True):
        super().__init__()
        self.pad, self.dim, self.width = pad, len(grid), width
        padded = tuple(n + pad for n in grid)
        self.lift = nn.Linear(in_ch + self.dim, width)
        self.wave = nn.ModuleList(WaveConv(width, padded, level, wavelet, fine) for _ in range(layers))
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(layers)) if modes else None
        conv = nn.Conv1d if self.dim == 1 else nn.Conv2d
        self.skip = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, out_ch))

    def features(self, x):
        h = self.lift(torch.cat([x, grid_coords(x)], -1)).movedim(-1, 1)
        if self.pad:
            h = F.pad(h, [0, self.pad] * self.dim)
        for i in range(len(self.wave)):
            z = self.wave[i](h) + self.skip[i](h)
            if self.spectral is not None:
                z = z + self.spectral[i](h)
            h = F.gelu(z) if i < len(self.wave) - 1 else z
        if self.pad:
            h = h[(...,) + tuple(slice(0, n - self.pad) for n in h.shape[2:])]
        return h.movedim(1, -1)

    def forward(self, x):
        return self.proj(self.features(x))
