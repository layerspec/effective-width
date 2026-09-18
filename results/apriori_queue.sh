#!/bin/bash
# A-priori feasibility part A (2026-09-18): random-init profiles on Gaussian surrogate images.
# While the matched seed-1 pass runs, the P2 training process is paused (SIGSTOP) so the probe's many
# small MPS kernels do not queue behind the training kernels (7x slowdown measured); resumed afterwards.
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
pause_training STOP; echo "training paused $(date)"
trap 'pause_training CONT' EXIT
$PY scripts/apriori_check.py measure --inputs matched --seed 1 $M
pause_training CONT; echo "training resumed $(date)"
$PY scripts/apriori_check.py measure --inputs matched --seed 2 $M
$PY scripts/apriori_check.py measure --inputs pink --seed 1 $M
$PY scripts/apriori_check.py measure --inputs white --seed 1 $M
echo "=== apriori queue done $(date) ==="
