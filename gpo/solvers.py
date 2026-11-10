"""Iterative solvers for (Ka (x) Kx + Psi Psi^T + s2 I) alpha = R without forming the (N d) x (N d) matrix.

`sdd` is stochastic dual descent (Lin et al., 2023), the solver of the GPO paper: random blocks of fields, Nesterov
momentum, geometric iterate averaging. The products it needs (Psi^T alpha, alpha Kx and their momentum
counterparts) are maintained incrementally, so one step costs O(B N d + B d^2 + N d) and memory stays O(N d) per
right-hand side. `cg` is conjugate gradients with the full matrix-vector product, O(N^2 d + N d^2) per iteration.
"""
import torch


class KernelOperator:
    """Matrix-free K = Ka (x) Kx + Psi Psi^T + s2 I.

    Psi is either a tensor (N, d, c) or a callable rows -> Psi[rows] that recomputes features on demand, so that
    the N x d x c feature tensor never has to be stored.
    """

    def __init__(self, Ka, Kx, s2, Psi, n_feat=None, chunk=256):
        self.Ka, self.Kx, self.s2, self.Psi, self.chunk = Ka, Kx, s2, Psi, chunk
        self.N, self.d = Ka.shape[0], Kx.shape[0]
        self.c = Psi.shape[-1] if torch.is_tensor(Psi) else (n_feat or 0)

    def rows(self, I=None):
        if self.Psi is None:
            n = self.N if I is None else len(I)
            return torch.zeros(n, self.d, 0, dtype=self.Ka.dtype, device=self.Ka.device)
        if torch.is_tensor(self.Psi):
            return self.Psi if I is None else self.Psi[I]
        return self.Psi(torch.arange(self.N, device=self.Ka.device) if I is None else I)

    def _blocks(self):
        for i in range(0, self.N, self.chunk):
            I = torch.arange(i, min(i + self.chunk, self.N), device=self.Ka.device)
            yield I, self.rows(I)

    def psi_t(self, V, rows=None):
        """Psi^T V for V (..., n, d) on the given rows (all rows by default), returns (..., c)."""
        if self.c == 0:
            return V.new_zeros(*V.shape[:-2], 0)
        if rows is None and callable(self.Psi):
            return sum(self.psi_t(V[..., I, :], I) for I, _ in self._blocks())
        P = self.rows(rows)
        out = V.reshape(-1, V.shape[-2] * V.shape[-1]).to(P.dtype) @ P.reshape(-1, P.shape[-1])
        return out.reshape(*V.shape[:-2], P.shape[-1])

    def psi(self, w, rows=None):
        """Psi w for w (..., c), returns (..., n, d)."""
        if self.c == 0:
            return w.new_zeros(*w.shape[:-1], self.N if rows is None else len(rows), self.d)
        if rows is None and callable(self.Psi):
            return torch.cat([self.psi(w, I) for I, _ in self._blocks()], -2)
        P = self.rows(rows)
        n = P.shape[0]
        return (P.reshape(-1, P.shape[-1]) @ w.reshape(-1, w.shape[-1]).to(P.dtype).T).T.reshape(
            *w.shape[:-1], n, self.d).to(self.Ka.dtype)

    def gram(self):
        """Psi^T Psi, (c, c)."""
        if torch.is_tensor(self.Psi):
            F = self.Psi.reshape(-1, self.c)
            return (F.T @ F).to(self.Ka.dtype)
        G = 0
        for _, P in self._blocks():
            F = P.reshape(-1, self.c).to(self.Ka.dtype)
            G = G + F.T @ F
        return G

    def kron_part(self):
        return KernelOperator(self.Ka, self.Kx, self.s2, None, chunk=self.chunk)

    def matvec(self, V):
        return self.psi(self.psi_t(V).to(self.Ka.dtype)) + self.Ka @ V @ self.Kx + self.s2 * V

    def lambda_max(self, iters=30, generator=None):
        if getattr(self, "_lam", None) is not None:
            return self._lam
        v = torch.randn(self.N, self.d, dtype=self.Ka.dtype, device=self.Ka.device, generator=generator)
        for _ in range(iters):
            v = self.matvec(v)
            lam = v.norm()
            v = v / lam
        self._lam = lam.item()
        return self._lam


@torch.no_grad()
def cg(K, R, iters=500, tol=1e-6, callback=None):
    """Conjugate gradients for K X = R with R (..., N, d); one system per leading index."""
    X = torch.zeros_like(R)
    r = R.clone()
    p = r.clone()
    dims = (-2, -1)
    rs = (r * r).sum(dims, keepdim=True)
    r0 = rs.sqrt()
    for it in range(iters):
        Kp = K.matvec(p)
        a = rs / (p * Kp).sum(dims, keepdim=True)
        X = X + a * p
        r = r - a * Kp
        rs_new = (r * r).sum(dims, keepdim=True)
        if callback is not None:
            callback(it, X)
        if (rs_new.sqrt() / r0).max() < tol:
            break
        p = r + rs_new / rs * p
        rs = rs_new
    return X


@torch.no_grad()
def sdd(K, R, steps=10_000, batch=64, lr=0.5, momentum=0.9, averaging=None, nesterov=True, generator=None,
        callback=None, every=100, X0=None, backfit=0):
    """Stochastic dual descent for K X = R, R (N, d) or (S, N, d).

    Minimises the dual objective 0.5 X^T K X - X^T R with block-coordinate gradients on `batch` random fields,
    step size lr / lambda_max(K), Nesterov momentum and geometric averaging X_bar <- r X + (1 - r) X_bar.
    Returns X_bar and the averaged products (Psi^T X_bar, X_bar Kx) needed for predictions.
    """
    if backfit:
        return _backfit(K, R, backfit, steps, batch=batch, lr=lr, momentum=momentum, averaging=averaging,
                        nesterov=nesterov, generator=generator, callback=callback)
    squeeze = R.dim() == 2
    R = R[None] if squeeze else R
    S, N, d = R.shape
    dt, dev = K.Ka.dtype, K.Ka.device
    r = averaging if averaging is not None else min(1.0, 100.0 / steps)
    beta = lr / K.lambda_max(generator=generator)
    X, V, Xb = (torch.zeros(S, N, d, dtype=dt, device=dev) for _ in range(3))
    PX, PV, PXb = (torch.zeros(S, K.c, dtype=dt, device=dev) for _ in range(3))
    XK, VK, XbK = (torch.zeros(S, N, d, dtype=dt, device=dev) for _ in range(3))
    if X0 is not None:
        X0 = X0[None] if X0.dim() == 2 else X0
        X.copy_(X0); Xb.copy_(X0)
        PX.copy_(K.psi_t(X0).to(dt)); PXb.copy_(PX)
        XK.copy_(X0 @ K.Kx); XbK.copy_(XK)
    scale = N / batch
    for t in range(steps):
        I = torch.randperm(N, device=dev, generator=generator)[:batch]
        mu = momentum if nesterov else 0.0
        U_I = X[:, I] + mu * V[:, I]
        PU = PX + mu * PV
        UK = XK + mu * VK
        G = scale * (K.psi(PU, I) + torch.matmul(K.Ka[I], UK) + K.s2 * U_I - R[:, I])
        V.mul_(momentum); PV.mul_(momentum); VK.mul_(momentum)
        V[:, I] -= beta * G
        PV -= beta * K.psi_t(G, I).to(dt)
        VK[:, I] -= beta * (G @ K.Kx)
        X += V; PX += PV; XK += VK
        Xb.mul_(1 - r).add_(r * X); PXb.mul_(1 - r).add_(r * PX); XbK.mul_(1 - r).add_(r * XK)
        if callback is not None and (t % every == 0 or t == steps - 1):
            callback(t, Xb[0] if squeeze else Xb)
    if squeeze:
        return Xb[0], PXb[0], XbK[0]
    return Xb, PXb, XbK


def _backfit(K, R, outer, steps, callback=None, **kw):
    """Joint posterior mean of y = Psi w + g + eps by Gauss-Seidel over the two blocks.

    The linear block is solved exactly in weight space, w = (Psi^T Psi + s2 I)^-1 Psi^T (R - g), and the Kronecker
    block g = (Ka (x) Kx) S^-1 (R - Psi w) by warm-started SDD on S = Ka (x) Kx + s2 I, whose spectrum is not dominated
    by the rank-c feature block. At the fixed point alpha = S^-1 (R - Psi w) = K^-1 R and w = Psi^T alpha.
    Returns alpha, w and alpha Kx.
    """
    squeeze = R.dim() == 2
    R = R[None] if squeeze else R
    Kk = K.kron_part()
    G = K.gram()
    L = torch.linalg.cholesky(G + K.s2 * torch.eye(K.c, dtype=G.dtype, device=G.device))
    w_step = lambda T: torch.cholesky_solve(K.psi_t(T).to(G.dtype).T, L).T
    A = torch.zeros_like(R)
    w = w_step(R)
    per = max(1, steps // (outer + 1))
    for it in range(outer + 1):
        r2 = R - K.psi(w)
        A, _, AK = sdd(Kk, r2, steps=per, X0=A, **kw)
        A, AK = (A[None], AK[None]) if A.dim() == 2 else (A, AK)
        if it < outer:
            w = w_step(R - (r2 - K.s2 * A))
        if callback is not None:
            callback(it, A[0] if squeeze else A, w[0] if squeeze else w)
    if squeeze:
        return A[0], w[0], AK[0]
    return A, w, AK
