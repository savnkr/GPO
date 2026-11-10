import math
import torch

from gpo.resample import fourier, interpolate, nearest, to_grid


def test_fourier_is_exact_for_band_limited_fields():
    n, m = 64, 256
    xf = torch.arange(m) / m
    f = lambda t: torch.sin(2 * math.pi * 3 * t) + 0.5 * torch.cos(2 * math.pi * 7 * t)
    fine = f(xf)[None, :, None]
    coarse = fine[:, :: m // n]
    torch.testing.assert_close(fourier(fine, (n,)), coarse, atol=1e-5, rtol=0)
    torch.testing.assert_close(fourier(coarse, (m,)), fine, atol=1e-5, rtol=0)


def test_fourier_2d_round_trip():
    g = torch.Generator().manual_seed(0)
    c = fourier(torch.randn(1, 12, 12, 1, generator=g), (16, 16))
    torch.testing.assert_close(fourier(fourier(c, (32, 32)), (16, 16)), c, atol=1e-5, rtol=0)


def test_nested_grids_are_subsampled():
    x = torch.randn(2, 800, 1)
    torch.testing.assert_close(to_grid(x, (200,), periodic=True, smooth=False), x[:, ::4])
    y = torch.randn(1, 421, 421, 1)
    torch.testing.assert_close(to_grid(y, (85, 85), periodic=False), y[:, ::5, ::5])


def test_periodic_linear_wraps():
    x = torch.tensor([0.0, 1.0, 2.0, 1.0])[None, :, None]
    out = interpolate(x, (8,), periodic=True, mode="linear")[0, :, 0]
    torch.testing.assert_close(out, torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 1.5, 1.0, 0.5]))


def test_bounded_upsampling_keeps_smooth_fields():
    tc, tf = torch.linspace(0, 1, 43), torch.linspace(0, 1, 85)
    g = lambda t: torch.sin(3 * t)[:, None] * torch.cos(2 * t)[None]
    out = to_grid(g(tc)[None, ..., None], (85, 85), periodic=False)[0, ..., 0]
    torch.testing.assert_close(out, g(tf), atol=1e-2, rtol=0)


def test_identity_on_training_grid():
    x = torch.randn(2, 10, 1)
    assert to_grid(x, (10,), periodic=True) is x


def test_nearest_keeps_values():
    x = torch.tensor([0.0, 0.0, 2.0, 2.0, 0.0])[None, :, None]
    out = to_grid(x, (10,), periodic=True, smooth=False)[0, :, 0]
    assert set(out.tolist()) <= {0.0, 2.0}
    torch.testing.assert_close(nearest(x, (5,), True), x)


def test_fourier_odd_sizes():
    for n, m in [(15, 45), (45, 15), (16, 45)]:
        xf = torch.arange(m) / m
        xc = torch.arange(n) / n
        f = lambda t: torch.sin(2 * math.pi * 3 * t) + torch.cos(2 * math.pi * 5 * t)
        torch.testing.assert_close(fourier(f(xc)[None, :, None], (m,))[0, :, 0], f(xf), atol=1e-4, rtol=0)
