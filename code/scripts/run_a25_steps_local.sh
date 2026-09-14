#!/usr/bin/env bash
# A25 (analysis-plan 9.13): CIFAR-10 at 640 images per class, 780 epochs = 39,000 steps (the full set's step count),
# three seeds, then the post-ReLU widths on the same 6,400 training images.  Resume-safe, skips finished runs.
# Usage: cd /Users/ccli/Downloads/effective-width-claude && nohup caffeinate -i code/scripts/run_a25_steps_local.sh >> results/a25_steps.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/effwidth/bin/python}
OUT=../results/a25_steps
mkdir -p $OUT
for s in 0 1 2; do
  name=c10_pc640_e780_s$s
  if [ -f "$OUT/$name/epoch_780_layers.csv" ]; then echo "=== $name already done, skipping ==="; else
    echo "=== $name start $(date) ==="
    $PY scripts/trajectory_cifar.py --device mps --epochs 780 --schedule 780 --workers 0 --gpu-data --data-root ../data \
      --arch vgg16_bn --dataset cifar10 --per-class 640 --seed $s --out $OUT/$name > $OUT/$name.log 2>&1
    echo "=== $name end $(date) ==="
  fi
  if [ ! -f "$OUT/$name/widths_act.json" ] && [ -f "$OUT/$name/vgg16_bn_cifar10_final.pt" ]; then
    echo "=== widths $name $(date) ==="
    $PY scripts/width_from_ruler.py --dataset cifar10 --per-class 640 --tensor act --checkpoint $OUT/$name/vgg16_bn_cifar10_final.pt \
      --data ../data --device mps --n-images 6400 --out $OUT/$name/widths_act.json > $OUT/$name/widths_act.log 2>&1
  fi
done
echo "=== A25 done $(date) ==="
