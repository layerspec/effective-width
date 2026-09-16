#!/bin/bash
# Mac queue 0917c: zero-retraining block-internal fold on ImageNet ResNets; DeiT-S eps rerun.
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=~/.venvs/effwidth/bin/python
echo "=== mac queue 0917c start $(date) ==="
for m in resnet18 resnet50 timm:resnet50.a1_in1k timm:resnet50.tv2_in1k; do
  stem=$(echo "$m" | sed 's/:/_/')
  [ -f ../results/cssp_resnet/${stem}_fold.csv ] && { echo "skip $m"; continue; }
  echo "--- fold_resnet $m $(date)"
  $PY scripts/fold_resnet.py --model "$m" --device mps --workers 2 --out ../results/cssp_resnet
done
echo "--- fold_mlp deit_small (eps rerun) $(date)"
$PY scripts/fold_mlp.py --model deit_small_patch16_224.fb_in1k --device mps --workers 2 --out ../results/cssp_mlp
echo "=== mac queue 0917c done $(date) ==="
