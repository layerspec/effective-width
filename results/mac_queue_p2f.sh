#!/bin/bash
# P2f (2026-09-20): the P2d wrong-start net [16,16,40,40,96,96,96,317x6] trained AS IS (no controller), three seeds —
# the missing comparison arm for P2d. Waits for mac_queue_p2e.sh. Resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2f.sh >> results/mac_queue_p2f.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e)\.sh' > /dev/null; do sleep 120; done
while pgrep -f 'scripts/trajectory_cifar.py' > /dev/null; do sleep 120; done
echo "=== mac_queue_p2f start/resume $(date) ==="
NARROW="16 16 40 40 96 96 96 317 317 317 317 317 317"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
for s in 0 1 2; do run ../results/p2/c10_narrow341_asis_s$s --dataset cifar10 --seed $s --widths $NARROW; done
echo "=== mac_queue_p2f all done $(date) ==="
