#!/bin/bash
# Queued 2026-09-07 after trajectory_queue.sh: four more same-dataset checkpoints
# for the inverted-residual families on the 6,400-image subset.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f trajectory_queue.sh > /dev/null; do sleep 120; done
echo "=== inverted-residual extra checkpoints start $(date) ==="
$PY -m layerspec.run --data ../data/imagenet_val_6400 --device mps --batch-size 32 --workers 4 --positions 16 --seed 0 \
  --models timm:tf_efficientnet_b0.in1k timm:tf_efficientnet_b0.aa_in1k timm:tf_efficientnet_b0.ap_in1k timm:tf_mobilenetv3_large_100.in1k \
  --out ../results/local6400/archs/trained
echo "=== measure queue done $(date) ==="
