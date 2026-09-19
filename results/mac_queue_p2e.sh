#!/bin/bash
# P2e grow-cap sweep (2026-09-20): every P2/P2b/b345/C100 run grew the front sites to the 1.25x cap, so the cap binds.
# Same setting as P2 (CIFAR-10, budget 0.673) with --realloc-grow-cap 0.5 and 1.0. Order: cap 0.5 s0, cap 1.0 s0 (trend by
# Sunday night), then cap 0.5 s1, s2 (statistics; resumable). Waits for mac_queue_p2d.sh.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2e.sh >> results/mac_queue_p2e.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_(0918|p2c|p2d)\.sh' > /dev/null; do sleep 120; done
while pgrep -f 'scripts/trajectory_cifar.py' > /dev/null; do sleep 120; done
echo "=== mac_queue_p2e start/resume $(date) ==="
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
run ../results/p2/c10_b673_cap50_s0  --dataset cifar10 --seed 0 --realloc-budget 0.673 --realloc-grow-cap 0.5
run ../results/p2/c10_b673_cap100_s0 --dataset cifar10 --seed 0 --realloc-budget 0.673 --realloc-grow-cap 1.0
run ../results/p2/c10_b673_cap50_s1  --dataset cifar10 --seed 1 --realloc-budget 0.673 --realloc-grow-cap 0.5
run ../results/p2/c10_b673_cap50_s2  --dataset cifar10 --seed 2 --realloc-budget 0.673 --realloc-grow-cap 0.5
echo "=== mac_queue_p2e all done $(date) ==="
