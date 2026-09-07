#!/bin/bash
# Queued 2026-09-07 after decompose_queue.sh: the projection anchor on two more
# architectures (submission bar A10), same protocol as the ResNet-50 run.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'decompose_queue.sh' > /dev/null; do sleep 120; done
for m in vgg16_bn resnet18 timm:resnet50.a1_in1k; do
  echo "=== projection $m start $(date) ==="
  $PY scripts/projection_check.py --model $m --data ../data/imagenet_val_6400 --device mps --positions 64 --taus 0.9 0.95 0.99 0.999 --out ../results/projection
done
echo "=== projection queue done $(date) ==="
