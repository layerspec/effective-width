#!/bin/bash
# A16: numerical-rank tolerance ablation for the Proposition-1 check (ResNet-50 V1), 1e-4 and 1e-8.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
for tol in 1e-4 1e-8; do
  echo "=== tolerance $tol $(date) ==="
  $PY scripts/decompose_check.py --model resnet50 --data ../data/imagenet_val_6400 --device mps --positions 48 --workers 2 --rank-tol $tol --out ../results/decompose_tol$tol
done
echo "=== tolerance queue done $(date) ==="
