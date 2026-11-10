"""Exact linear algebra for K = Ka (x) Kx + Psi Psi^T + s2 I on N fields of d grid points.

Ka (N, N) is the input kernel, Kx (d, d) the spatial kernel and Psi (N, d, c) the scaled operator features of the
linear kernel. S = Ka (x) Kx + s2 I is diagonalised by Qa (x) Qx; the rank-c part is handled with Woodbury, so
solves, log-determinants and marginal variances cost O(N^3 + d^3 + c N d (N + d)) instead of O((N d)^3).
"""
import torch


def eigh(K):
    """Symmetric eigendecomposition with a CPU / jitter fallback when the GPU solver does not converge."""
    try:
        la, Q = torch.linalg.eigh(K)
    except torch.linalg.LinAlgError:
        eye = torch.eye(len(K), dtype=K.dtype)
        for jitter in (0.0, 1e-8, 1e-6, 1e-4):
            try:
                la, Q = torch.linalg.eigh(K.cpu() + jitter * K.diagonal().mean().item() * eye)
                la, Q = la.to(K.device), Q.to(K.device)
                break
            except torch.linalg.LinAlgError:
                continue
        else:
            raise
    return la.clamp_min(0), Q


class KronEig:
    """Eigen-factorisation of S = Ka (x) Kx + s2 I."""

    def __init__(self, Ka, Kx, s2, eig_a=None, eig_x=None):
        self.la, self.Qa = eig_a if eig_a is not None else eigh(Ka)
        self.lx, self.Qx = eig_x if eig_x is not None else eigh(Kx)
        self.s2 = s2
        self.lam = self.la[:, None] * self.lx[None] + s2

    def to_eig(self, V):
        return self.Qa.T @ V @ self.Qx

    def from_eig(self, V):
        return self.Qa @ V @ self.Qx.T

    def solve(self, V):
        """S^-1 V for V (..., N, d)."""
        return self.from_eig(self.to_eig(V) / self.lam)

    def logdet(self):
        return self.lam.log().sum()


class Woodbury:
    """K = S + Psi Psi^T: solves, log-determinant, and the quantities shared by gradients and predictions."""

    def __init__(self, kron, Psi, chunk=None):
        self.kron, self.Psi = kron, Psi.contiguous()
        N, d, c = Psi.shape
        chunk = chunk or c
        self.P = torch.empty_like(self.Psi)                                     # S^-1 Psi, (N, d, c)
        for k in range(0, c, chunk):
            self.P[..., k:k + chunk] = kron.solve(self.Psi[..., k:k + chunk].movedim(-1, 0)).movedim(0, -1)
        self.E = self.flat(self.Psi).T @ self.flat(self.P)                     # Psi^T S^-1 Psi
        self.E = 0.5 * (self.E + self.E.T)
        self.M = self.E + torch.eye(c, dtype=Psi.dtype, device=Psi.device)
        self.Lm = torch.linalg.cholesky(self.M)

    @staticmethod
    def flat(T):
        return T.reshape(-1, T.shape[-1])

    def solve(self, R):
        """K^-1 R for R (..., N, d)."""
        SR = self.kron.solve(R)
        lead = SR.shape[:-2]
        SRf = SR.reshape(-1, SR.shape[-2] * SR.shape[-1])
        t = torch.cholesky_solve((SRf @ self.flat(self.Psi)).T, self.Lm)
        return (SRf - (self.flat(self.P) @ t).T).reshape(*lead, *SR.shape[-2:])

    def logdet(self):
        return self.kron.logdet() + 2 * self.Lm.diagonal().log().sum()


class ExactNLL(torch.autograd.Function):
    """0.5 y^T K^-1 y + 0.5 log|K| with closed-form gradients in (Ka, Kx, s2, Psi, Y).

    With G = dL/dK = 0.5 (K^-1 - a a^T), a = K^-1 y and K^-1 = S^-1 - P M^-1 P^T, every Kronecker contraction of G
    is evaluated in the eigenbasis or through the rank-c factors; no derivative of eigh is needed, so repeated
    eigenvalues of grid kernels are harmless.
    """

    @staticmethod
    def forward(ctx, Ka, Kx, s2, Psi, Y):
        kron = KronEig(Ka, Kx, s2)
        wb = Woodbury(kron, Psi)
        A = wb.solve(Y)
        ctx.save_for_backward(Ka, Kx, Psi, A)
        ctx.wb = wb
        return 0.5 * (Y * A).sum() + 0.5 * wb.logdet()

    @staticmethod
    def backward(ctx, g):
        Ka, Kx, Psi, A = ctx.saved_tensors
        wb = ctx.wb
        kron, P = wb.kron, wb.P
        R = torch.cholesky_solve(P.reshape(-1, P.shape[-1]).T, wb.Lm).T.reshape(P.shape)  # P M^-1 = K^-1 Psi
        dKa = dKx = ds2 = dPsi = dY = None
        if ctx.needs_input_grad[0]:
            PKx = torch.einsum("st,mtk->msk", Kx, P)
            dKa = 0.5 * ((kron.Qa * (kron.lx[None] / kron.lam).sum(1)[None]) @ kron.Qa.T
                         - torch.einsum("nsk,msk->nm", R, PKx) - A @ Kx @ A.T)
        if ctx.needs_input_grad[1]:
            KaP = torch.einsum("nm,mtk->ntk", Ka, P)
            dKx = 0.5 * ((kron.Qx * (kron.la[:, None] / kron.lam).sum(0)[None]) @ kron.Qx.T
                         - torch.einsum("nsk,ntk->st", R, KaP) - A.T @ Ka @ A)
        if ctx.needs_input_grad[2]:
            trKinv = (1 / kron.lam).sum() - torch.einsum("ndk,ndk->", R, P)
            ds2 = 0.5 * (trKinv - (A ** 2).sum())
        if ctx.needs_input_grad[3]:
            dPsi = R - A[..., None] * (A.reshape(1, -1) @ Psi.reshape(-1, Psi.shape[-1]))
        if ctx.needs_input_grad[4]:
            dY = A
        ctx.wb = None
        return (None if dKa is None else g * dKa, None if dKx is None else g * dKx,
                None if ds2 is None else g * ds2, None if dPsi is None else g * dPsi, None if dY is None else g * dY)


def exact_nll(Ka, Kx, s2, Psi, Y):
    return ExactNLL.apply(Ka, Kx, s2, Psi, Y)
