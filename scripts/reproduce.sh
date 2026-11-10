#!/usr/bin/env bash
# GPO on every benchmark (3 seeds), ablations, theory checks, applications, tables and figures.
# Assumes the stage-1 operators in checkpoints/<bench>_s<seed> (scripts/train_operators.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
BENCH="burgers advection_width advection darcy darcy_notch navier_stokes"
AMPLITUDE="advection_width advection darcy_notch"

for b in $BENCH; do
  extra=""
  [[ " $AMPLITUDE " == *" $b "* ]] && extra="--amplitude --heldout-cond 256"
  for s in 0 1 2; do
    $PY experiments/train.py --bench $b --init checkpoints/${b}_s$s --out runs/gpo/${b}_s$s --seed $s $extra
  done
  $PY experiments/train.py --bench $b --init checkpoints/${b}_s0 --out runs/ablation/${b}_paper_kernel --no-linear --no-spatial --heldout-steps 0
  $PY experiments/train.py --bench $b --init checkpoints/${b}_s0 --out runs/ablation/${b}_paper_kernel_joint --no-linear --no-spatial \
    --lr-op 3e-4 --steps 1500 --eval-every 100 --heldout-steps 0
  $PY experiments/train.py --bench $b --init checkpoints/${b}_s0 --out runs/ablation/${b}_no_linear --no-linear --heldout-steps 0
done

for b in $BENCH; do
  PY=$PY bash scripts/analysis.sh $b
done

$PY scripts/collect.py --runs runs/gpo --ablation runs/ablation
$PY figures/matheron_toy.py --out assets/media
$PY figures/plots.py --out assets/figures
$PY scripts/build_page.py
