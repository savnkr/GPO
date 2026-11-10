# Data

All loaders read from `$GPO_DATA` (default `./datasets`). Three benchmarks are official files and three are generated;
the notched-domain generator needs `triangle` and `scikit-fem` (`pip install triangle scikit-fem`).

```bash
bash scripts/data/download.sh                                                        # Burgers, Darcy, Navier-Stokes
python scripts/data/gen_advection.py --ic 1 --n 3000 --nx 200 --nt 101 --seed 0 --out datasets/gen/advection_ic1_200
python scripts/data/gen_advection.py --ic 2 --n 3000 --nx 200 --nt 101 --seed 0 --out datasets/gen/advection_ic2_200
python scripts/data/gen_darcy_notch.py --n 4000 --hmax 0.03 --seed 0 --out datasets/gen/darcy_notch
```

| Benchmark | Map | Grid | Train / val / test | Source |
|---|---|---|---|---|
| Burgers | u(x, 0) → u(x, 1), ν = 0.1 | 1024 | 1900 / 48 / 100 | Li et al. (2021), `burgers_data_R10.mat` |
| Wave advection, varying width | u(x, 0) → u(x, 0.5) | 200 | 2000 / 200 / 100 | Lu et al. (2022) IC1, generated; the paper's case |
| Wave advection, wave and bump | u(x, 0) → u(x, 0.5) | 200 | 2000 / 200 / 100 | Lu et al. (2022) IC2, generated |
| Darcy flow | a(x) → u(x) | 85 × 85 | 1000 / 200 / 200 | Li et al. (2021), `piececonst_r421_N1024_smooth{1,2}.mat` |
| Darcy, notched domain | boundary condition → u(x) | 51 × 51 | 1500 / 200 / 100 | Lu et al. (2022), generated |
| Navier–Stokes | forcing → vorticity at T = 10 | 64 × 64 | 3000 / 200 / 100 | de Hoop et al. (2022), 40 000 pairs |

Darcy uses the standard split (train on the first 1000 fields of `smooth1`, test on the first 200 of `smooth2`), so
numbers are directly comparable with published operator-learning results. The remaining 36 700 Navier–Stokes pairs,
never seen in training, are used for the data-scaling experiment.

```
datasets/
  raw/caltech/NavierStokes_{inputs,outputs}.npy
  raw/fno/burgers/burgers_data_R10.mat
  raw/hf_mutransfer/piececonst_r421_N1024_smooth1.mat
  raw/fno/darcy421/piececonst_r421_N1024_smooth2.mat
  gen/advection_ic1_200/advection_ic1.npz
  gen/advection_ic2_200/advection_ic2.npz
  gen/darcy_notch/darcy_notch.npz
```
