#!/bin/bash
# 2026-10-01 night (plan 9.27; author: the Mac stays on, run all three): after p2l,
#   (A) P-b7.1'  memory-window Gram on the three A18 ruler-width runs (CPU, ~2 h)
#   (C) P9.85/86 CIFAR-10-image profiles of the six random families, seeds 2 and 3 (~30 min)
#   (B) P-b7.4   weight-decay sweep, VGG-16_BN CIFAR-10 full width, 3 lambdas x 2 seeds (MPS, ~1.4 h each) + window measurement
# Resumable: finished runs / measurements are skipped.
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'results/mac_queue_p2[l]\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
until pgrep -f 'scripts/trajectory_(cifar|vit|r20wide)\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_queue_1001 start $(date) ==="
# (A)
for s in 0 1 2; do
  [ -f ../results/gradwindow/vgg16_bn_ruler_s${s}_layers.csv ] || \
    $PY -u scripts/measure_gradwindow.py --run ../results/a18_pt/vgg16_bn_ruler_s$s --lam 5e-4 --out ../results/gradwindow --device cpu
done
$PY scripts/measure_gradwindow.py --judge ruler --out ../results/gradwindow
echo "=== (A) done $(date) ==="
# (C)
M="resnet50_random timm:resnet18.tv_in1k_random timm:vgg16_bn.tv_in1k_random timm:densenet121.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random"
for s in 2 3; do
  $PY -u scripts/apriori_check.py --dataset cifar10=../data/cifar10_png_6400 measure --inputs cifar10 --seed $s $M
done
$PY scripts/l1_forgetting_judge.py --results ../results
echo "=== (C) done $(date) ==="
# (B)
for s in 0 1; do for wd in 5e-5 5e-4 5e-3; do
  out=../results/wd_sweep/wd${wd}_s$s
  [ -f $out/epoch_100_layers.csv ] || \
    $PY -u scripts/trajectory_wd.py --wd $wd --arch vgg16_bn --dataset cifar10 --seed $s --device mps --save-checkpoints \
        --schedule 0 40 50 60 70 80 90 100 --out $out
  [ -f ../results/gradwindow/wd${wd}_s${s}_layers.csv ] || \
    $PY -u scripts/measure_gradwindow.py --run $out --lam $wd --out ../results/gradwindow --device cpu
done; done
$PY scripts/measure_gradwindow.py --judge sweep --out ../results/gradwindow
echo "=== (B) done; mac_queue_1001 done $(date) ==="
