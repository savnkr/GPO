"""Maps an input function given on any grid to the grid the model was trained on.

A finer grid that contains the training grid is subsampled, which is exactly how benchmark inputs at the training
resolution are produced, for smooth and discontinuous inputs alike. Otherwise smooth inputs are resampled in Fourier
space (periodic domains) or bicubically (bounded domains, sampled at both end points), and discontinuous inputs take the
nearest sample, which keeps a piecewise-constant input piecewise constant (no ringing, no ramps the model never saw).
"""
import itertools
import torch
import torch.nn.functional as F


def fourier(x, size):
    """x (b, *grid, c) periodic, sampled without the end point -> (b, *size, c)."""
    dims = tuple(range(1, x.dim() - 1))
    X = torch.fft.fftn(x, dim=dims, norm="forward")
    out = torch.zeros(x.shape[0], *size, x.shape[-1], dtype=X.dtype, device=x.device)
    sl_in, sl_out = [], []
    for n, m in zip(x.shape[1:-1], size):
        kpos, kneg = min(n, m) // 2, (min(n, m) - 1) // 2
        sl_in.append((slice(0, kpos + 1), slice(n - kneg, n)))
        sl_out.append((slice(0, kpos + 1), slice(m - kneg, m)))
    for parts in itertools.product(*[range(2)] * len(size)):
        src = tuple(sl_in[a][p] for a, p in enumerate(parts))
        dst = tuple(sl_out[a][p] for a, p in enumerate(parts))
        out[(slice(None),) + dst] = X[(slice(None),) + src]
    return torch.fft.ifftn(out, dim=dims, norm="forward").real.to(x.dtype)


def interpolate(x, size, periodic, mode):
    """Linear or cubic interpolation of x (b, *grid, c); periodic grids exclude the end point, bounded ones include it."""
    t = x.movedim(-1, 1)
    if periodic:
        t = torch.cat([t, t[:, :, :1]], 2)
        if t.dim() == 4:
            t = torch.cat([t, t[..., :1]], 3)
        size_full = tuple(m + 1 for m in size)
    else:
        size_full = tuple(size)
    if t.dim() == 3:
        out = F.interpolate(t[..., None], size=(size_full[0], 1), mode="bilinear" if mode == "linear" else "bicubic",
                            align_corners=True)[..., 0]
    else:
        out = F.interpolate(t, size=size_full, mode="bilinear" if mode == "linear" else "bicubic", align_corners=True)
    if periodic:
        out = out[(slice(None), slice(None)) + tuple(slice(0, m) for m in size)]
    return out.movedim(1, -1)


def nearest(x, size, periodic):
    """Nearest-sample resampling of x (b, *grid, c)."""
    idx = []
    for n, m in zip(x.shape[1:-1], size):
        j = torch.arange(m, device=x.device, dtype=torch.float64)
        idx.append(torch.round(j * n / m).long() % n if periodic else torch.round(j * (n - 1) / (m - 1)).long())
    for a, i in enumerate(idx):
        x = x.index_select(a + 1, i)
    return x


def nested_stride(grid, size, periodic):
    """Stride s with grid[::s] == size along every axis, or None."""
    strides = []
    for n, m in zip(grid, size):
        if periodic and n % m == 0:
            strides.append(n // m)
        elif not periodic and (n - 1) % (m - 1) == 0:
            strides.append((n - 1) // (m - 1))
        else:
            return None
    return strides


def to_grid(x, size, periodic, smooth=True):
    """Input field x (b, *grid, c) on any grid -> the training grid `size`."""
    grid = tuple(x.shape[1:-1])
    if grid == tuple(size):
        return x
    s = nested_stride(grid, size, periodic)
    if s is not None:
        return x[(slice(None),) + tuple(slice(None, None, k) for k in s)]
    if smooth:
        return fourier(x, size) if periodic else interpolate(x, size, False, "cubic")
    return nearest(x, size, periodic)
