import math
import pytest
import torch

from gpo.kernels import matern52
from gpo.linalg import KronEig, Woodbury, exact_nll

torch.set_default_dtype(torch.float64)


def problem(N=5, d=6, c=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    za, zx = torch.randn(N, 2, generator=g), torch.rand(d, 1, generator=g)
    Ka, Kx = 0.7 * matern52(za, za, 1.3), matern52(zx, zx, 0.4)
    Psi = 0.3 * torch.randn(N, d, c, generator=g)
    Y = torch.randn(N, d, generator=g)
    return Ka, Kx, torch.tensor(0.05), Psi, Y


def dense(Ka, Kx, s2, Psi):
    F = Psi.reshape(-1, Psi.shape[-1])
    return torch.kron(Ka, Kx) + F @ F.T + s2 * torch.eye(F.shape[0])


def dense_nll(Ka, Kx, s2, Psi, Y):
    K = dense(Ka, Kx, s2, Psi)
    y = Y.reshape(-1)
    return 0.5 * y @ torch.linalg.solve(K, y) + 0.5 * torch.logdet(K)


def test_solve_and_logdet():
    Ka, Kx, s2, Psi, Y = problem()
    wb = Woodbury(KronEig(Ka, Kx, s2), Psi)
    K = dense(Ka, Kx, s2, Psi)
    torch.testing.assert_close(wb.solve(Y).reshape(-1), torch.linalg.solve(K, Y.reshape(-1)))
    torch.testing.assert_close(wb.logdet(), torch.logdet(K))


def test_batched_solve():
    Ka, Kx, s2, Psi, Y = problem()
    wb = Woodbury(KronEig(Ka, Kx, s2), Psi)
    R = torch.randn(4, *Y.shape)
    torch.testing.assert_close(wb.solve(R), torch.stack([wb.solve(r) for r in R]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nll_gradients(seed):
    args = [t.clone().requires_grad_() for t in problem(seed=seed)]
    exact_nll(*args).backward()
    ref = [t.detach().clone().requires_grad_() for t in args]
    val = dense_nll(*ref)
    val.backward()
    torch.testing.assert_close(exact_nll(*[t.detach() for t in args]), val.detach())
    for a, r in zip(args, ref):
        ga, gr = a.grad, r.grad
        if a.dim() == 2 and a.shape[0] == a.shape[1]:
            ga, gr = 0.5 * (ga + ga.T), 0.5 * (gr + gr.T)
        torch.testing.assert_close(ga, gr, rtol=1e-6, atol=1e-8)


def test_repeated_eigenvalues():
    """A regular grid kernel on a torus has exactly repeated eigenvalues; gradients must stay finite."""
    x = torch.linspace(0, 1, 9)[:-1, None]
    Kx = torch.cos(2 * math.pi * (x - x.T))
    Kx = Kx + 1.5 * torch.eye(8)
    Ka, _, s2, Psi, Y = problem(d=8)
    Kx = Kx.requires_grad_()
    exact_nll(Ka, Kx, s2, Psi, Y).backward()
    assert torch.isfinite(Kx.grad).all()
