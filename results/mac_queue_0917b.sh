#!/bin/bash
# Mac queue 0917b: zero-retraining MLP-width fold on pretrained transformers / ConvNeXt (waits for queue 0917).
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=~/.venvs/effwidth/bin/python
while pgrep -f mac_queue_0917.sh > /dev/null; do sleep 60; done
echo "=== mac queue 0917b start $(date) ==="
for m in deit_small_patch16_224.fb_in1k vit_base_patch16_224.augreg_in21k_ft_in1k convnext_tiny.fb_in1k; do
  [ -f ../results/cssp_mlp/${m}_fold.csv ] && { echo "skip $m"; continue; }
  echo "--- fold_mlp $m $(date)"
  $PY scripts/fold_mlp.py --model $m --device mps --workers 2 --out ../results/cssp_mlp
done
echo "=== mac queue 0917b done $(date) ==="
