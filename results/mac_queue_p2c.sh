#!/bin/bash
# P2c ablation (2026-09-19): P2 without the grow operator (--realloc-grow-cap 0, shrink-only water-filling at the
# same 0.673 budget), three seeds, so the contribution of "grow" can be separated from "shrink" in the review.
# Waits for the 0918 chain, then runs; resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2c.sh >> results/mac_queue_p2c.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_0918.sh' > /dev/null; do sleep 120; done
while pgrep -f 'scripts/trajectory_cifar.py' > /dev/null; do sleep 120; done
echo "=== mac_queue_p2c start/resume $(date) ==="
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
for s in 0 1 2; do run ../results/p2/c10_b673_nogrow_s$s --dataset cifar10 --seed $s --realloc-budget 0.673 --realloc-grow-cap 0; done
echo "=== mac_queue_p2c all done $(date) ==="
