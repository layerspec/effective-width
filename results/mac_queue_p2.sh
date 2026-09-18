#!/bin/bash
# P2 / P3 (2026-09-17): one training run with the width controller (layerspec.realloc) against A18's
# three arms.  VGG-16 (BN), 100 epochs, controller acts after epochs 15 (grow+shrink) and 40 (shrink),
# budget = A18's ruler-arm parameter fraction (0.673 = post-activation ruler, 0.345 = conv-output ruler).
# Resumable: relaunch daily; finished runs are skipped.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2.sh >> results/mac_queue_p2.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_(0917|r20)' > /dev/null; do sleep 60; done
echo "=== mac_queue_p2 start/resume $(date) ==="
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn "$@" --out $out
}
# Order (2026-09-18 night, 48-hour unattended window, ~3 h per run when the GPU is not shared):
#   1. c10_b673 seeds 1-2   -> P2 with three seeds (arms: A18 full 93.77, ruler 93.84)
#   2. c10_fixed339 seeds 0-2 (P2b, conserved budget from the uniform 0.339 start; arms: uniform 93.20, ruler 93.55)
#   3. c10_b345 seeds 0-2   (aggressive budget; arms: ruler-0.345 93.55, uniform 93.20)
#   4. c100_b673 seeds 0-2  (CIFAR-100 has no tau=0.999 slack; stress test)
UNIFORM="37 37 75 75 149 149 149 298 298 298 298 298 298"
for s in 0 1 2; do run ../results/p2/c10_b673_s$s --dataset cifar10  --seed $s --realloc-budget 0.673; done
for s in 0 1 2; do run ../results/p2/c10_fixed339_s$s --dataset cifar10 --seed $s --widths $UNIFORM --realloc-budget 0.339; done
for s in 0 1 2; do run ../results/p2/c10_b345_s$s --dataset cifar10  --seed $s --realloc-budget 0.345; done
for s in 0 1 2; do run ../results/p2/c100_b673_s$s --dataset cifar100 --seed $s --realloc-budget 0.673; done
echo "=== mac_queue_p2 all done $(date) ==="
