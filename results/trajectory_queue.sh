#!/bin/bash
# Queued 2026-09-07 on the Mac (MPS). Log: results/trajectory_queue.log
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
echo "=== resnet50 seed 0 start $(date) ==="
$PY scripts/trajectory_cifar.py --arch resnet50 --device mps --epochs 100 --workers 4 --seed 0 --out ../results/trajectory_resnet50
echo "=== vgg16_bn seed 1 start $(date) ==="
$PY scripts/trajectory_cifar.py --arch vgg16_bn --device mps --epochs 100 --workers 4 --seed 1 --out ../results/trajectory_seed1
echo "=== vgg16_bn seed 2 start $(date) ==="
$PY scripts/trajectory_cifar.py --arch vgg16_bn --device mps --epochs 100 --workers 4 --seed 2 --out ../results/trajectory_seed2
echo "=== queue done $(date) ==="
