#!/bin/bash
# P2g (2026-09-20, hostile-review route A): the controller conserving FORWARD MACs instead of parameters
# (--realloc-cost flops). Budgets are the MACs fractions of the existing A18 arms so every comparison is at equal
# compute: 0.88 = ruler_act (276M of 313M), 0.60 = ruler (189M), 0.342 = uniform (107M; P2b-style start from the
# uniform net at conserved MACs). Seed 0 of each group first, then seeds 1 and 2. Waits for mac_queue_p2f.sh.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2g.sh >> results/mac_queue_p2g.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
until pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e|p2f)\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # wait unless pgrep says 'no match'; a failing pgrep (shutdown) counts as busy
until pgrep -f 'scripts/trajectory_cifar.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # wait unless pgrep says 'no match'; a failing pgrep (shutdown) counts as busy
echo "=== mac_queue_p2g start/resume $(date) ==="
UNIFORM="37 37 75 75 149 149 149 298 298 298 298 298 298"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn --realloc-cost flops "$@" --out $out
}
for s in 0 1 2; do
  run ../results/p2/c10_f88_s$s       --dataset cifar10 --seed $s --realloc-budget 0.88
  run ../results/p2/c10_f60_s$s       --dataset cifar10 --seed $s --realloc-budget 0.60
  run ../results/p2/c10_ffixed342_s$s --dataset cifar10 --seed $s --widths $UNIFORM --realloc-budget 0.342
done
echo "=== mac_queue_p2g all done $(date) ==="
