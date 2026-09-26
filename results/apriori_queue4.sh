#!/bin/bash
# A7 (2026-09-26, apriori-feasibility note): a different-spectrum image set (CIFAR-10 test images upsampled to 224).
# Random-init profiles on the real CIFAR-10 images and on a Gaussian surrogate with CIFAR-10's own spectrum, seed 1,
# six families; compared with the ImageNet-spectrum surrogate (matched, already measured) and the real ImageNet profile.
# Pauses the training process while measuring (shared MPS is 7x slower), resumes it afterwards.
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
M="resnet50_random timm:resnet18.tv_in1k_random timm:vgg16_bn.tv_in1k_random timm:densenet121.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random"
pause_training() {
  for p in $(pgrep -f "scripts/trajectory_(cifar|vit|r20wide)\.py"); do
    kill -$1 $p; for c in $(pgrep -P $p); do kill -$1 $c; done
  done
}
echo "=== apriori queue4 start $(date) ==="
pause_training STOP; echo "training paused $(date)"
trap 'pause_training CONT' EXIT
$PY -u scripts/apriori_check.py --dataset cifar10=../data/cifar10_png_6400 measure --inputs cifar10 --seed 1 $M
$PY -u scripts/apriori_check.py --dataset cifar10=../data/cifar10_png_6400 measure --inputs cifar10matched --seed 1 $M
pause_training CONT; echo "training resumed $(date)"
$PY scripts/apriori_check.py --dataset cifar10=../data/cifar10_png_6400 compare --reference cifar10 --inputs cifar10matched matched imagenet pink white
echo "=== apriori queue4 done $(date) ==="
