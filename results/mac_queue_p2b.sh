#!/bin/bash
# P2b (2026-09-17, the author's "fixed lanes" framing): the hardware is given, so hold the total parameter
# count and only REALLOCATE.  Start from A18's uniform-cut VGG-16 (0.339 of full) and let the controller
# move channels between layers at conserved budget; compare with the uniform arm trained as is (93.20)
# and with the ruler arm (93.55).  Waits for mac_queue_p2.sh.
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_p2.sh' > /dev/null; do sleep 60; done
echo "=== mac_queue_p2b start/resume $(date) ==="
UNIFORM="37 37 75 75 149 149 149 298 298 298 298 298 298"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
for s in 0 1 2; do run ../results/p2/c10_fixed339_s$s --dataset cifar10 --seed $s --widths $UNIFORM --realloc-budget 0.339; done
echo "=== mac_queue_p2b all done $(date) ==="
