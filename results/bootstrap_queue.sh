#!/bin/bash
# D1 / M3: block bootstrap over images (hostile review 2026-09-15, table S13).
# CPU only, 6 threads, runs beside the MPS training queue. Resumable: collect
# skips when meta.json exists, bootstrap skips when replicates.csv exists.
# Block statistics (~6 GB for ResNet-50) are deleted once replicates.csv is written.
cd "$(dirname "$0")/../code" || exit 1
PY=~/.venvs/effwidth/bin/python
OUT=../results/bootstrap
echo "=== bootstrap queue start $(date) ==="
for m in resnet50 resnet50_random vgg16_bn vgg16_bn_random; do
  echo "=== $m collect $(date) ==="
  $PY scripts/bootstrap_images.py collect --model $m --data ../data/imagenet_val_6400 --out $OUT \
      --block-images 200 --batch-size 50 --positions 16 --patch-positions 48 --workers 2 --threads 6 || exit 1
  echo "=== $m bootstrap $(date) ==="
  $PY scripts/bootstrap_images.py bootstrap --model $m --out $OUT --reps 200 --threads 6 || exit 1
  if [ -f $OUT/$m/replicates.csv ] && [ -f $OUT/$m/align.csv ]; then
    rm -f $OUT/$m/M2_*.npy $OUT/$m/pooled_M2_*.npy
  fi
done
echo "=== summarise $(date) ==="
$PY scripts/bootstrap_images.py summarise --out $OUT --trained resnet50 vgg16_bn \
    --random resnet50_random vgg16_bn_random --latex ../paper/table_bootstrap.tex
echo "=== bootstrap queue done $(date) ==="
