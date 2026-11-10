## Accuracy

| Benchmark | GPO (paper) | **GPO (this release)** | Improvement | WNO (paper) | GP (paper) |
|---|---|---|---|---|---|
| Burgers | 2.89 ± 0.11 | **0.92 ± 0.10** | 3.1× | 4.09 | 4.91 |
| Wave advection, varying width | 0.63 ± 0.08 | **0.06 ± 0.02** | 11.4× | 1.01 | 1.24 |
| Wave advection, wave and bump | – | **0.33 ± 0.05** | – | – | – |
| Darcy flow | 3.98 ± 0.12 | **0.52 ± 0.00** | 7.7× | 2.82 | 6.27 |
| Darcy, notched domain | 2.18 ± 0.19 | **0.16 ± 0.03** | 14.0× | 2.21 | 5.22 |
| Navier–Stokes | 2.21 ± 0.44 | **0.32 ± 0.01** | 6.9× | 12.23 | 9.12 |

## Uncertainty

| Benchmark | NLL ↓ | CRPS ↓ | 95% coverage | 95% coverage (no recalibration) | ρ(σ, error) per field |
|---|---|---|---|---|---|
| Burgers | -4.09 ± 0.09 | 2.52e-03 | 0.93 ± 0.02 | 0.91 ± 0.03 | 0.84 ± 0.01 |
| Wave advection, varying width | -6.41 ± 0.30 | 2.81e-04 | 0.93 ± 0.00 | 0.90 ± 0.02 | 0.61 ± 0.06 |
| Wave advection, wave and bump | -5.08 ± 0.14 | 1.22e-03 | 0.95 ± 0.00 | 0.92 ± 0.00 | 0.83 ± 0.05 |
| Darcy flow | -8.66 ± 0.01 | 2.03e-05 | 0.93 ± 0.00 | 0.88 ± 0.00 | 0.36 ± 0.05 |
| Darcy, notched domain | -5.38 ± 0.09 | 7.11e-04 | 0.95 ± 0.00 | 0.94 ± 0.00 | 0.63 ± 0.07 |
| Navier–Stokes | -7.35 ± 0.09 | 7.46e-05 | 0.95 ± 0.00 | 0.99 ± 0.00 | 0.63 ± 0.20 |

## Ablation

| Benchmark | input-embedding kernel, operator frozen | input-embedding kernel, trained jointly | + spatial kernel | + operator-feature kernel (GPO) |
|---|---|---|---|---|
| Burgers | 7.72 | 3.48 | 6.65 | 0.92 |
| Wave advection, varying width | 11.88 | 9.36 | 11.90 | 0.06 |
| Wave advection, wave and bump | 14.87 | 10.02 | 14.23 | 0.33 |
| Darcy flow | 25.59 | 6.77 | 25.59 | 0.52 |
| Darcy, notched domain | 6.60 | 1.86 | 4.50 | 0.16 |
| Navier–Stokes | 100.03 | 100.05 | 90.23 | 0.32 |
