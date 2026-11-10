#!/usr/bin/env bash
# Applications, theory checks and media for one trained run: scripts/analysis.sh <bench> [run_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
b=$1
run=${2:-runs/gpo/${b}_s0}
mkdir -p results/apps results/theory assets/media
$PY experiments/sensor_fusion.py --run $run --out results/apps/sensors_$b.pt
$PY experiments/functionals.py --run $run --out results/apps/functionals_$b.pt
$PY figures/media.py --run $run --out assets/media
$PY figures/media.py --run $run --sensors results/apps/sensors_$b.pt --out assets/media
$PY experiments/resolution.py --run $run --out results/apps/resolution_$b.pt
if [ "$b" = navier_stokes ] || [ "$b" = burgers ]; then
  $PY experiments/theory.py --run $run --out results/theory/$b.pt
fi
if [ "$b" = navier_stokes ]; then
  $PY experiments/data_scaling.py --run $run --out results/apps/data_scaling_navier_stokes.pt
fi
