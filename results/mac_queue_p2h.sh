#!/bin/bash
# P2h (2026-09-20 evening, hostile-review routes A' and A''):
#   A'  apriori673:   the controller acts on the UNTRAINED network (epoch 0) as well as at 15/40 — the a-priori start
#                      (route 2 used by route 1 for the first time); budget 0.673 params, compare with P2 (94.11) and P2d.
#   A'' p2widths:     P2 seed-0's final widths trained from scratch, no controller — is the value in the widths or the controller?
#   A'' randgrow:     P2 with grow along random complement directions — does the unmet-variance direction matter?
# Seed 0 of each first, then seeds 1 and 2. Waits for mac_queue_p2g.sh. Resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2h.sh >> results/mac_queue_p2h.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e|p2f|p2g)\.sh' > /dev/null; do sleep 120; done
while pgrep -f 'scripts/trajectory_cifar.py' > /dev/null; do sleep 120; done
echo "=== mac_queue_p2h start/resume $(date) ==="
P2W="64 80 159 160 320 320 320 632 402 435 236 164 170"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
for s in 0 1 2; do
  run ../results/p2/c10_apriori673_s$s --dataset cifar10 --seed $s --realloc-budget 0.673 --realloc-epochs 0 15 40
  run ../results/p2/c10_p2widths_s$s   --dataset cifar10 --seed $s --widths $P2W
  run ../results/p2/c10_b673_randgrow_s$s --dataset cifar10 --seed $s --realloc-budget 0.673 --realloc-grow-random
done
echo "=== mac_queue_p2h all done $(date) ==="
