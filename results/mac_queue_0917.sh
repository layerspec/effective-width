#!/bin/bash
# Mac queue, 2026-09-17: no-new-code jobs while the grow operator is being written.
#  1. P1 fold on the other two CIFAR-10 full-width seeds (error bars)
#  2. P1 fold on the three CIFAR-100 full-width seeds (P3's zero-retraining half)
#  3. Sigma_patch decompositions of random initialisations for seven families (a-priori profile groundwork)
#  4. per-layer CSSP table for CIFAR-100 seed 0
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=~/.venvs/effwidth/bin/python
echo "=== mac queue 0917 start $(date) ==="
for s in 1 2; do
  out=../results/cssp/seed$s
  [ -f $out/fold_results.csv ] && { echo "skip cifar10 seed $s"; continue; }
  echo "--- fold cifar10 seed $s $(date)"
  $PY scripts/fold_cssp.py --device mps --checkpoint ../results/round3_pt/a5_vgg16_bn_none_s$s/vgg16_bn_cifar10_final.pt --out $out
done
for s in 0 1 2; do
  out=../results/cssp/cifar100_s$s
  [ -f $out/fold_results.csv ] && { echo "skip cifar100 seed $s"; continue; }
  echo "--- fold cifar100 seed $s $(date)"
  $PY scripts/fold_cssp.py --device mps --dataset cifar100 --widths-act ../results/a18_cifar100/widths_act.json \
      --checkpoint ../results/a18_pt/cifar100/vgg16_bn_full_s$s/vgg16_bn_cifar100_final.pt --out $out
done
for m in resnet18_random vgg16_bn_random timm:resnet34.a1_in1k_random timm:densenet121.tv_in1k_random \
         timm:mobilenetv3_large_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random timm:resnet50.a1_in1k_random; do
  stem=$(echo "$m" | sed 's/timm:/timm_/')
  [ -f ../results/decompose/${stem}_decompose.csv ] && { echo "skip $m"; continue; }
  echo "--- decompose $m $(date)"
  $PY scripts/decompose_check.py --model "$m" --data ../data/imagenet_val_6400 --device mps --positions 48 --workers 2 --out ../results/decompose
done
out=../results/cssp/cifar100_s0
[ -f $out/cssp_per_layer.csv ] || { echo "--- cssp per-layer cifar100 s0 $(date)"; $PY scripts/cssp_check.py --device mps --dataset cifar100 --widths-act ../results/a18_cifar100/widths_act.json --checkpoint ../results/a18_pt/cifar100/vgg16_bn_full_s0/vgg16_bn_cifar100_final.pt --out $out; }
echo "=== mac queue 0917 done $(date) ==="
