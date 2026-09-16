#!/bin/bash
# Multi-day Mac queue (2026-09-16): (1) random-init patch-covariance decompositions, seeds 1-2, seven families;
# (2) full-width ResNet-20 baselines on CIFAR-10 and CIFAR-100, three seeds each (Yuan et al. 2023's CIFAR
# network; baselines for the controller's P2/P3 and for the a-priori profile on a third block family).
# Safe to relaunch daily: decompositions skip if their CSV exists, training resumes from resume.pt.
#   cd <repo> && nohup caffeinate -i results/mac_queue_r20.sh >> results/mac_queue_r20.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_0917' > /dev/null; do sleep 60; done
echo "=== mac_queue_r20 start/resume $(date) ==="
for seed in 1 2; do
  for m in resnet18_random vgg16_bn_random timm:resnet34.a1_in1k_random timm:densenet121.tv_in1k_random \
           timm:mobilenetv3_large_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random timm:resnet50.a1_in1k_random; do
    stem=$(echo "$m" | sed 's/timm:/timm_/'); out=../results/decompose_seed$seed
    [ -f $out/${stem}_decompose.csv ] && continue
    echo "--- decompose $m seed $seed $(date)"
    $PY scripts/decompose_check.py --model "$m" --seed $seed --data ../data/imagenet_val_6400 --device mps --positions 48 --workers 2 --out $out
  done
done
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch resnet20 "$@" --out $out
}
for s in 0 1 2; do run ../results/r20/cifar10_full_s$s --dataset cifar10 --seed $s; done
for s in 0 1 2; do run ../results/r20/cifar100_full_s$s --dataset cifar100 --seed $s; done
echo "=== mac_queue_r20 all done $(date) ==="
