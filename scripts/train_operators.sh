#!/usr/bin/env bash
# Stage 1: trains the operator of every benchmark for seeds 0, 1, 2 into checkpoints/<bench>_s<seed>.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

for s in 0 1 2; do
  $PY experiments/train_operator.py --bench burgers --arch wno --width 64 --level 8 --layers 4 --wavelet db6 \
    --epochs 500 --batch 20 --lr 1e-3 --wd 1e-4 --h1 0.1 --aug shift --seed $s --out checkpoints/burgers_s$s
  $PY experiments/train_operator.py --bench advection --arch ewno --width 64 --level 3 --layers 4 --wavelet db4 \
    --modes 32 --epochs 500 --batch 32 --lr 1e-3 --wd 1e-4 --h1 0.1 --aug shift --seed $s --out checkpoints/advection_s$s
  $PY experiments/train_operator.py --bench advection_width --arch ewno --width 64 --level 3 --layers 4 --wavelet db4 \
    --modes 32 --epochs 500 --batch 32 --lr 1e-3 --wd 1e-4 --h1 0.1 --aug shift --seed $s --out checkpoints/advection_width_s$s
  $PY experiments/train_operator.py --bench darcy --arch wnof --width 64 --level 2 --layers 4 --wavelet db6 \
    --modes 12 12 --epochs 500 --batch 20 --lr 1e-3 --wd 1e-4 --h1 0.1 --aug d4 --seed $s --out checkpoints/darcy_s$s
  $PY experiments/train_operator.py --bench darcy_notch --arch wno --width 64 --level 2 --layers 4 --wavelet db6 \
    --epochs 300 --batch 16 --lr 5e-4 --wd 1e-4 --h1 0.1 --aug none --seed $s --out checkpoints/darcy_notch_s$s
  $PY experiments/train_operator.py --bench navier_stokes --arch wnof --width 48 --level 2 --layers 4 --wavelet db6 \
    --modes 16 16 --epochs 200 --batch 32 --lr 3e-4 --wd 1e-4 --h1 0.1 --aug none --seed $s --out checkpoints/navier_stokes_s$s
done
