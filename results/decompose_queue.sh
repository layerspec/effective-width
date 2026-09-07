#!/bin/bash
# Queued 2026-09-07: kernel-vs-data decomposition on the 6,400-image subset.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
for m in resnet50 vgg16_bn resnet18 timm:resnet50.a1_in1k timm:resnet50.tv2_in1k timm:resnet34.a1_in1k timm:densenet121.tv_in1k timm:mobilenetv3_large_100.ra_in1k timm:efficientnet_b0.ra_in1k resnet50_random; do
  echo "=== $m start $(date) ==="
  $PY scripts/decompose_check.py --model $m --data ../data/imagenet_val_6400 --device mps --positions 48 --workers 4 --out ../results/decompose
done
echo "=== decompose queue done $(date) ==="
