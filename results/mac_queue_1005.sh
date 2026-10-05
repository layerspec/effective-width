#!/bin/bash
# 2026-10-05 (plan 9.30; the Mac is idle, the author: run 1 + 2 and 3 overnight):
#   (A) B9 with the number of positions changed: VGG-16_BN CIFAR-10 at 48 px, seeds 0 and 1 (MPS), then the cell measurement and the judge
#   (B) P-b7.4d': weight decay 1e-3 and 2e-3, two seeds (MPS) + window measurement, then the level judge
# Resumable: finished runs / measurements are skipped.
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
echo "=== mac_queue_1005 start $(date) ==="
for s in 0 1; do
  out=../results/position_sweep/px48_s$s
  [ -f $out/epoch_100_layers.csv ] || $PY -u scripts/position_sweep.py train --px 48 --seed $s --device mps --out $out
  $PY -u scripts/position_sweep.py measure --px 48 --run $out --name px48_s$s
  $PY scripts/position_sweep.py judge --results ../results
done
echo "=== (A) done $(date) ==="
for s in 0 1; do for wd in 1e-3 2e-3; do
  out=../results/wd_sweep/wd${wd}_s$s
  [ -f $out/epoch_100_layers.csv ] || \
    $PY -u scripts/trajectory_wd.py --wd $wd --arch vgg16_bn --dataset cifar10 --seed $s --device mps --save-checkpoints \
        --schedule 0 40 50 60 70 80 90 100 --out $out
  [ -f ../results/gradwindow/wd${wd}_s${s}_layers.csv ] || \
    $PY -u scripts/measure_gradwindow.py --run $out --lam $wd --out ../results/gradwindow --device cpu
done; done
$PY scripts/wd_level_judge.py --results ../results
echo "=== (B) done; mac_queue_1005 done $(date) ==="
