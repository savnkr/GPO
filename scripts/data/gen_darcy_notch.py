"""Darcy (Poisson) on a triangle with a notch (Lu et al. 2022, lu-group/deeponet-fno data/darcy_triangular_notch).

-Lap u = 10 on the triangle (0,0), (1,0), (0.5, sqrt(3)/2) minus the notch [0.49, 0.51] x [0, 0.4].
u = g(x) on the outer edges, with g ~ GP(0, exp(-(x-x')^2 / (2 * 0.2^2))) sampled on 101 points and
interpolated linearly in x; u = 0 on the notch. P1 finite elements on a quality mesh with max
edge about `hmax`, then interpolation to a 101 x 101 grid, linear inside and nearest outside,
as in the MATLAB original. Saves boundCoeff (g repeated along y), sol, and the domain mask.

    python scripts/data/gen_darcy_notch.py --n 2000 --out DIR
"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import triangle as tr
import skfem
from skfem.helpers import dot, grad
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

S3 = np.sqrt(3) / 2
NL, NR, NH = 0.49, 0.51, 0.4


def build_mesh(hmax):
    v = np.array([[0, 0], [NL, 0], [NL, NH], [NR, NH], [NR, 0], [1, 0], [0.5, S3]])
    seg = np.array([[i, (i + 1) % len(v)] for i in range(len(v))])
    m = tr.triangulate({"vertices": v, "segments": seg}, f"pq30a{hmax ** 2 * np.sqrt(3) / 4:.8f}")
    mesh = skfem.MeshTri(m["vertices"].T, m["triangles"].T)
    p = mesh.p.T
    on_notch = (np.abs(p[:, 0] - NL) < 1e-9) | (np.abs(p[:, 0] - NR) < 1e-9) | \
               ((np.abs(p[:, 1] - NH) < 1e-9) & (p[:, 0] >= NL - 1e-9) & (p[:, 0] <= NR + 1e-9))
    return mesh, on_notch


@skfem.BilinearForm
def lap(u, v, _):
    return dot(grad(u), grad(v))


@skfem.LinearForm
def rhs(v, _):
    return 10.0 * v


def setup(hmax):
    mesh, on_notch = build_mesh(hmax)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    A, b = lap.assemble(basis), rhs.assemble(basis)
    bnd = mesh.boundary_nodes()
    return mesh, A, b, bnd, on_notch[bnd]


STATE = {}


def init(hmax):
    STATE["s"] = setup(hmax)


def solve_one(g):
    mesh, A, b, bnd, notch = STATE["s"]
    xf = np.linspace(0, 1, len(g))
    ub = np.where(notch, 0.0, np.interp(mesh.p[0, bnd], xf, g))
    u = np.zeros(mesh.p.shape[1]); u[bnd] = ub
    u = skfem.solve(*skfem.condense(A, b, x=u, D=bnd))
    pts = mesh.p.T
    xq, yq = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    q = np.c_[xq.ravel(), yq.ravel()]
    v = LinearNDInterpolator(pts, u)(q)
    nan = np.isnan(v)
    v[nan] = NearestNDInterpolator(pts, u)(q[nan])
    return v.reshape(101, 101).astype(np.float32)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=2000); p.add_argument("--hmax", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    xf = np.linspace(0, 1, 101)
    C = np.exp(-0.5 * (xf[:, None] - xf[None]) ** 2 / 0.2 ** 2)
    g = np.random.default_rng(a.seed).multivariate_normal(np.zeros(101), C, a.n, method="eigh")
    t0 = time.time()
    with ProcessPoolExecutor(a.workers, initializer=init, initargs=(a.hmax,)) as ex:
        sol = np.stack(list(ex.map(solve_one, g, chunksize=16)))
    xq, yq = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
    inside = (yq <= S3 * (1 - np.abs(2 * xq - 1)) + 1e-12) & ~((xq > NL) & (xq < NR) & (yq < NH))
    os.makedirs(a.out, exist_ok=True)
    np.savez(f"{a.out}/darcy_notch.npz", boundCoeff=np.repeat(g[:, None, :], 101, 1).astype(np.float32),
             sol=sol, f_bc=g.astype(np.float32), mask=inside, coord_x=xq, coord_y=yq)
    json.dump(vars(a), open(f"{a.out}/meta.json", "w"), indent=1)
    print(sol.shape, f"{time.time() - t0:.0f}s")
