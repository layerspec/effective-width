#!/usr/bin/env bash
# A24, class-count control (analysis-plan 9.12): VGG-16_BN on nested CIFAR-100 class subsets
# K = 13 / 20 / 50 (three seeds each) plus CIFAR-10 at 640 images per class (three seeds), on the Mac.
# (K = 13 and 640/class are the smallest that clear the n/C >= 50 gate at the 2x2 layers: 6,400 images x 4 positions / 512.)
# K = 100 is the CIFAR-100 full-width arm of plan 9.7 (results/a18_pt/cifar100/vgg16_bn_full_s*), not retrained.
# Then the post-ReLU widths of every final network on its own training subset (all seeds).
# Resume-safe (trajectory_cifar.py resume.pt) and skips finished runs.
# Usage: cd /Users/ccli/Downloads/effective-width-claude && nohup caffeinate -i code/scripts/run_a24_classes_local.sh >> results/a24_classes.log 2>&1 &
set -u
cd "$(dirname "$0")/.."            # code/
PY=${PY:-$HOME/.venvs/effwidth/bin/python}
OUT=../results/a24_classes
mkdir -p $OUT
COMMON="--device mps --epochs 100 --schedule 100 --workers 0 --gpu-data --data-root ../data --arch vgg16_bn"

run() {  # name, then trajectory_cifar.py args
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_100_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_cifar.py $COMMON "$@" --out $OUT/$name > $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
for s in 0 1 2; do
  for k in 13 20 50; do run c100_k${k}_s$s --dataset cifar100 --classes $k --seed $s; done
  run c10_pc640_s$s --dataset cifar10 --per-class 640 --seed $s
done

ruler() {  # name dataset extra-args checkpoint
  local name=$1 ds=$2 extra=$3 ckpt=$4
  if [ -f "$OUT/$name/widths_act.json" ]; then return; fi
  [ -f "$ckpt" ] || { echo "missing $ckpt"; return; }
  mkdir -p $OUT/$name
  echo "=== widths $name $(date) ==="
  $PY scripts/width_from_ruler.py --dataset $ds $extra --tensor act --checkpoint $ckpt --data ../data --device mps \
    --n-images 6400 --out $OUT/$name/widths_act.json > $OUT/$name/widths_act.log 2>&1
}
for s in 0 1 2; do
  for k in 13 20 50; do ruler c100_k${k}_s$s cifar100 "--classes $k" $OUT/c100_k${k}_s$s/vgg16_bn_cifar100_final.pt; done
  ruler c10_pc640_s$s cifar10 "--per-class 640" $OUT/c10_pc640_s$s/vgg16_bn_cifar10_final.pt
  ruler c100_k100_s$s cifar100 "" ../results/a18_pt/cifar100/vgg16_bn_full_s$s/vgg16_bn_cifar100_final.pt
  ruler c10_full_s$s cifar10 "" ../results/round3_pt/a5_vgg16_bn_none_s$s/vgg16_bn_cifar10_final.pt
done
echo "=== A24 done $(date) ==="
