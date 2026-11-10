"""Benchmark loaders. Each returns (x, y) as float32 tensors of shape (n, *grid, 1).

Files are read from $GPO_DATA (default: ./datasets); see data/README.md for sources.
"""
import os
import numpy as np
import scipy.io
import h5py
import torch

ROOT = os.environ.get("GPO_DATA", os.path.join(os.path.dirname(__file__), "..", "datasets"))
RAW, GEN = os.path.join(ROOT, "raw"), os.path.join(ROOT, "gen")


def read_mat(path, *fields):
    try:
        d = scipy.io.loadmat(path)
        return [torch.from_numpy(d[f].astype(np.float32)) for f in fields]
    except NotImplementedError:
        with h5py.File(path, "r") as d:
            return [torch.from_numpy(np.ascontiguousarray(d[f][()].T).astype(np.float32)) for f in fields]


def _npy(path):
    return np.load(path, mmap_mode="r", allow_pickle=False)


def burgers(sub=8):
    """Li et al. (2021): u0 -> u(T=1), nu = 0.1, 8192 points subsampled to 8192 / sub."""
    a, u = read_mat(f"{RAW}/fno/burgers/burgers_data_R10.mat", "a", "u")
    return a[:, ::sub, None], u[:, ::sub, None]


def advection(t_out=50, ic=2):
    """Lu et al. (2022) waves, u(x, 0) -> u(x, 0.5), 200 points: IC2 square wave plus bump, or IC1 square wave of
    random height, centre and width."""
    u = np.load(f"{GEN}/advection_ic{ic}_200/advection_ic{ic}.npz")["u"]
    return torch.from_numpy(u[:, 0].copy())[..., None], torch.from_numpy(u[:, t_out].copy())[..., None]


def darcy(sub=5):
    """Standard FNO Darcy split at 85x85: train smooth1[:1000], val smooth2[200:400], test smooth2[:200]."""
    c1, s1 = read_mat(f"{RAW}/hf_mutransfer/piececonst_r421_N1024_smooth1.mat", "coeff", "sol")
    c2, s2 = read_mat(f"{RAW}/fno/darcy421/piececonst_r421_N1024_smooth2.mat", "coeff", "sol")
    f = lambda t: t[:, ::sub, ::sub, None]
    return (torch.cat([f(c1)[:1000], f(c2)[200:400], f(c2)[:200]]),
            torch.cat([f(s1)[:1000], f(s2)[200:400], f(s2)[:200]]))


def darcy_notch(sub=2):
    """Lu et al. (2022) triangular domain with a notch: boundary condition -> pressure, 101x101 -> 51x51."""
    d = np.load(f"{GEN}/darcy_notch/darcy_notch.npz")
    x = torch.from_numpy(d["boundCoeff"][:, ::sub, ::sub].copy())
    y = torch.from_numpy(d["sol"][:, ::sub, ::sub].copy())
    return x[..., None], y[..., None]


def notch_mask(sub=2):
    return torch.from_numpy(np.load(f"{GEN}/darcy_notch/darcy_notch.npz")["mask"][::sub, ::sub].copy()).bool()


def navier_stokes(n=None):
    """de Hoop et al. (2022): forcing -> vorticity at T = 10, nu = 0.025, 64x64, 40000 pairs."""
    a = _npy(f"{RAW}/caltech/NavierStokes_inputs.npy")
    u = _npy(f"{RAW}/caltech/NavierStokes_outputs.npy")
    n = n or a.shape[-1]
    f = lambda v: torch.from_numpy(np.array(v[..., :n], dtype=np.float32)).permute(2, 1, 0)[..., None]
    return f(a), f(u)


BENCHMARKS = {
    "burgers": dict(load=burgers, ntrain=1900, nval=48, ntest=100, tail_test=True),
    "advection": dict(load=advection, ntrain=2000, nval=200, ntest=100, tail_test=False),
    "advection_width": dict(load=lambda: advection(ic=1), ntrain=2000, nval=200, ntest=100, tail_test=False),
    "darcy": dict(load=darcy, ntrain=1000, nval=200, ntest=200, tail_test=False),
    "darcy_notch": dict(load=darcy_notch, ntrain=1500, nval=200, ntest=100, tail_test=False),
    "navier_stokes": dict(load=lambda: navier_stokes(3300), ntrain=3000, nval=200, ntest=100, tail_test=False),
}


def load_split(name, ntrain=None, nval=None, ntest=None, tail_test=None):
    """Disjoint (train, val, test) pairs under the protocol of the benchmark; sizes can be overridden."""
    b = BENCHMARKS[name]
    ntrain, nval, ntest = ntrain or b["ntrain"], nval or b["nval"], ntest or b["ntest"]
    tail = b["tail_test"] if tail_test is None else tail_test
    x, y = b["load"]()
    tr, va = slice(0, ntrain), slice(ntrain, ntrain + nval)
    te = slice(len(x) - ntest, len(x)) if tail else slice(ntrain + nval, ntrain + nval + ntest)
    return (x[tr], y[tr]), (x[va], y[va]), (x[te], y[te])


def domain_mask(name):
    return notch_mask().flatten() if name == "darcy_notch" else None
