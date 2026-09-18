#!/bin/bash
# A-priori feasibility part A (2026-09-18): random-init profiles on Gaussian surrogate images.
# Waits for the P2 seed-0 run to reach epoch 100, then pauses the P2 training process (SIGSTOP) so the
# probe's many small MPS kernels do not queue behind the training kernels (7x slowdown measured), runs the
# ladder param / matched seed 2 / pink / white, and resumes training (also on exit).
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
M="resnet50_random timm:resnet18.tv_in1k_random timm:resnet34.tv_in1k_random timm:vgg16_bn.tv_in1k_random timm:densenet121.tv_in1k_random timm:resnet101.tv_in1k_random timm:wide_resnet50_2.tv_in1k_random timm:resnext50_32x4d.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random timm:mobilenetv3_large_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random"
pause_training() {
  for p in $(pgrep -f "scripts/trajectory_cifar.py"); do
    kill -$1 $p; for c in $(pgrep -P $p); do kill -$1 $c; done
  done
}
echo "=== apriori queue start $(date) ==="
until [ -f ../results/p2/c10_b673_s0/epoch_100_layers.csv ]; do sleep 30; done
pause_training STOP; echo "training paused $(date)"
trap 'pause_training CONT' EXIT
$PY -u scripts/apriori_check.py measure --inputs param --seed 1 $M
$PY -u scripts/apriori_check.py measure --inputs matched --seed 2 $M
$PY -u scripts/apriori_check.py measure --inputs pink --seed 1 $M
$PY -u scripts/apriori_check.py measure --inputs white --seed 1 $M
pause_training CONT; echo "training resumed $(date)"
echo "=== apriori queue done $(date) ==="
