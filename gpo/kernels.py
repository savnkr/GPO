import torch


def matern52(a, b, ell=1.0):
    """Matern-5/2 correlation between rows of a and b, length scale ell (scalar or per-dimension)."""
    s = 5 ** 0.5 * torch.cdist(a / ell, b / ell)
    return (1 + s + s ** 2 / 3) * torch.exp(-s)


def grid_coords(grid, device=None, dtype=torch.float64):
    axes = [torch.linspace(0, 1, n, device=device, dtype=dtype) for n in grid]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(grid))
