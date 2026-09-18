#!/bin/bash
# Phase-scrambled surrogate (2026-09-18 night, plan item 3): the real ImageNet-val images with per-image random
# Fourier phases (power spectrum and cross-spectrum kept exactly per image, higher-order structure destroyed).
# Waits for apriori_queue2.sh (COCO), pauses the P2 training process while measuring, resumes it afterwards.
#   cd <repo> && nohup caffeinate -i results/apriori_queue3.sh >> results/apriori_queue3.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
M="resnet50_random timm:resnet18.tv_in1k_random timm:resnet34.tv_in1k_random timm:vgg16_bn.tv_in1k_random timm:densenet121.tv_in1k_random timm:resnet101.tv_in1k_random timm:wide_resnet50_2.tv_in1k_random timm:resnext50_32x4d.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random timm:mobilenetv3_large_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random"
pause_training() {
  for p in $(pgrep -f "scripts/trajectory_cifar.py"); do
    kill -$1 $p; for c in $(pgrep -P $p); do kill -$1 $c; done
  done
}
while pgrep -f 'results/apriori_queue2.sh' > /dev/null; do sleep 60; done
echo "=== apriori queue3 (phase) start $(date) ==="
pause_training STOP; echo "training paused $(date)"
trap 'pause_training CONT' EXIT
$PY -u scripts/apriori_check.py measure --inputs phase --seed 1 $M
pause_training CONT; echo "training resumed $(date)"
echo "=== apriori queue3 done $(date) ==="
