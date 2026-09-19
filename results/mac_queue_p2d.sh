#!/bin/bash
# P2d (2026-09-19): P2b from a second, deliberately wrong start — front narrow, back wide
# [16,16,40,40,96,96,96,317x6] = 0.341 of the full parameters (uniform arm is 0.339), conserved budget, three seeds.
# Tests whether the controller corrects an arbitrary mis-allocation, not only the uniform one. Waits for mac_queue_p2c.sh.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2d.sh >> results/mac_queue_p2d.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_(0918|p2c)\.sh' > /dev/null; do sleep 120; done
while pgrep -f 'scripts/trajectory_cifar.py' > /dev/null; do sleep 120; done
echo "=== mac_queue_p2d start/resume $(date) ==="
NARROW="16 16 40 40 96 96 96 317 317 317 317 317 317"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
for s in 0 1 2; do run ../results/p2/c10_narrow341_s$s --dataset cifar10 --seed $s --widths $NARROW --realloc-budget 0.341; done
echo "=== mac_queue_p2d all done $(date) ==="
