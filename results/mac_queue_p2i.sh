#!/bin/bash
# P2i (2026-09-22, hostile review 09-20 "second weakest point: single architecture"): the controller on ResNet-20
# (block-internal widths, resnet_internal sites; residual stream 16/32/64 fixed), CIFAR-10, same recipe as the full-width
# baselines in results/r20/ (100 epochs, OneCycle, seeds 0-2: 92.56/92.38/92.38). FLOPs-conserving throughout.
#   f47        controller, budget 0.47 of full forward MACs, triggers 15/40 -> training cost ~0.55 of full, the training
#              cost of Yuan et al. (NeurIPS 2023) Table 2 (ResNet-20: 54.9% cost, 92.53 vs 92.62; their final net is full width)
#   ufixed47   uniform internal widths at 0.464 MACs [7 7 7 15 15 15 30 30 30], no controller: the equal-inference-compute control
#   apriori55  controller acting at epoch 0 as well (a-priori start), budget 0.55 -> training cost 0.55, inference 0.55
#   ufixed55   uniform at 0.552 MACs [9 9 9 17 17 17 35 35 35]: the control for apriori55
# Seed 0 of each first, then seeds 1 and 2 (~70 min per run). Waits for mac_queue_p2h.sh. Resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2i.sh >> results/mac_queue_p2i.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e|p2f|p2g|p2h)\.sh' > /dev/null; do sleep 120; done
while pgrep -f 'scripts/trajectory_cifar.py' > /dev/null; do sleep 120; done
echo "=== mac_queue_p2i start/resume $(date) ==="
U47="7 7 7 15 15 15 30 30 30"
U55="9 9 9 17 17 17 35 35 35"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch resnet20 --dataset cifar10 "$@" --out $out
}
for s in 0 1 2; do
  run ../results/p2/r20_f47_s$s       --seed $s --realloc-cost flops --realloc-budget 0.47
  run ../results/p2/r20_ufixed47_s$s  --seed $s --widths $U47
  run ../results/p2/r20_apriori55_s$s --seed $s --realloc-cost flops --realloc-budget 0.55 --realloc-epochs 0 15 40
  run ../results/p2/r20_ufixed55_s$s  --seed $s --widths $U55
done
echo "=== mac_queue_p2i all done $(date) ==="
