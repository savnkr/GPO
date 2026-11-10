"""Fourier neural operator (Li et al., 2021) on 1D/2D regular grids."""
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F


def grid_coords(x):
    """Unit-cube coordinates for a channel-last batch (b, *grid, c)."""
    axes = [torch.linspace(0, 1, n, device=x.device, dtype=x.dtype) for n in x.shape[1:-1]]
    g = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1)
    return g.expand(x.shape[0], *g.shape)


class SpectralConv(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.modes, self.dim = tuple(modes), len(modes)
        shape = (width, width, *modes)
        self.weights = nn.ParameterList(
            nn.Parameter(torch.randn(*shape, dtype=torch.cfloat) / width ** 2) for _ in range(2 ** (self.dim - 1)))

    def forward(self, x):
        dims = tuple(range(-self.dim, 0))
        ft = torch.fft.rfftn(x, dim=dims)
        out = torch.zeros_like(ft)
        ax = "xyz"[: self.dim]
        eq = f"bi{ax},io{ax}->bo{ax}"
        for w, signs in zip(self.weights, itertools.product((0, 1), repeat=self.dim - 1)):
            idx = tuple(slice(-m, None) if s else slice(0, m) for s, m in zip(signs, self.modes[:-1]))
            idx = idx + (slice(0, self.modes[-1]),)
            out[(..., *idx)] = torch.einsum(eq, ft[(..., *idx)], w)
        return torch.fft.irfftn(out, s=x.shape[-self.dim:], dim=dims)


class FNO(nn.Module):
    def __init__(self, modes, width=64, layers=4, in_ch=1, out_ch=1, proj=128, pad=0):
        super().__init__()
        self.pad, self.dim, self.width = pad, len(modes), width
        self.lift = nn.Linear(in_ch + self.dim, width)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(layers))
        conv = nn.Conv1d if self.dim == 1 else nn.Conv2d
        self.local = nn.ModuleList(conv(width, width, 1) for _ in range(layers))
        self.proj = nn.Sequential(nn.Linear(width, proj), nn.GELU(), nn.Linear(proj, out_ch))

    def features(self, x):
        h = self.lift(torch.cat([x, grid_coords(x)], -1)).movedim(-1, 1)
        if self.pad:
            h = F.pad(h, [0, self.pad] * self.dim)
        for i, (s, l) in enumerate(zip(self.spectral, self.local)):
            h = s(h) + l(h)
            if i < len(self.spectral) - 1:
                h = F.gelu(h)
        if self.pad:
            h = h[(...,) + tuple(slice(0, n - self.pad) for n in h.shape[2:])]
        return h.movedim(1, -1)

    def forward(self, x):
        return self.proj(self.features(x))
