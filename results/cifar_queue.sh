#!/bin/bash
# Replaces trajectory_queue.sh (2026-09-07 10:20): the ResNet-50 run died at
# epoch 7 with "Shared memory manager connection has timed out" while four
# queues shared the CPU.  This one waits until no other queue is running and
# uses two loader workers.  Log: results/cifar_queue.log
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'measure_queue.sh|measure_queue2.sh|decompose_queue.sh|projection_queue.sh' > /dev/null; do sleep 120; done
echo "=== resnet50 seed 0 start $(date) ==="
$PY scripts/trajectory_cifar.py --arch resnet50 --device mps --epochs 100 --workers 2 --seed 0 --out ../results/trajectory_resnet50
echo "=== vgg16_bn seed 1 start $(date) ==="
$PY scripts/trajectory_cifar.py --arch vgg16_bn --device mps --epochs 100 --workers 2 --seed 1 --out ../results/trajectory_seed1
echo "=== vgg16_bn seed 2 start $(date) ==="
$PY scripts/trajectory_cifar.py --arch vgg16_bn --device mps --epochs 100 --workers 2 --seed 2 --out ../results/trajectory_seed2
echo "=== cifar queue done $(date) ==="
