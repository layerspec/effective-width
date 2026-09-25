#!/bin/bash
# P2l (2026-09-25 evening; hostile review 09-25 items (e)1 and E1b).  Waits for mac_queue_p2k.sh.  Resumable.
#   1. c10_f60_nogrow_s{0,1,2}   VGG-16 CIFAR-10, FLOPs-conserving controller WITHOUT the grow operator
#                                (--realloc-grow-cap 0; shrink-only water-filling at 0.60 MACs): does growing on
#                                unmet-variance directions contribute under a fair (FLOPs) budget?  Compare with f60.
#   2. r20w4_full_s{0,1,2}       ResNet-20 with every stage x4 (64/128/256; 4.29M params, 644M MACs), full width:
#                                does idle width appear in the late layers when a narrow residual net is widened?
#   3. r20w4_f60_s{0,1,2}        the FLOPs-conserving controller on the wide net, 0.60 MACs, triggers 15/40
#   4. r20w4_ufixed60_s{0,1,2}   uniform block-internal widths at 0.600 MACs [38 38 38 77 77 77 154 154 154]
# ~2 h per VGG run, ~4 h per wide ResNet run on MPS: about 42 h.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2l.sh >> results/mac_queue_p2l.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
until pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e|p2f|p2g|p2h|p2i|p2j|p2k)\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep (shutdown) counts as busy
until pgrep -f 'scripts/trajectory_(cifar|vit|r20wide)\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_queue_p2l start/resume $(date) ==="
run() {  # out, script, then args
  local out=$1; local script=$2; shift 2
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/$script --device mps --epochs 100 --workers 2 --dataset cifar10 "$@" --out $out
}
CTRL="--realloc-budget 0.6 --realloc-cost flops"
for s in 0 1 2; do
  run ../results/p2/c10_f60_nogrow_s$s trajectory_cifar.py --arch vgg16_bn --seed $s $CTRL --realloc-grow-cap 0
done
run ../results/p2/r20w4_full_s0 trajectory_r20wide.py --width-mult 4 --seed 0
run ../results/p2/r20w4_f60_s0 trajectory_r20wide.py --width-mult 4 --seed 0 $CTRL
run ../results/p2/r20w4_ufixed60_s0 trajectory_r20wide.py --width-mult 4 --seed 0 --widths 38 38 38 77 77 77 154 154 154
for s in 1 2; do
  run ../results/p2/r20w4_full_s$s trajectory_r20wide.py --width-mult 4 --seed $s
  run ../results/p2/r20w4_f60_s$s trajectory_r20wide.py --width-mult 4 --seed $s $CTRL
  run ../results/p2/r20w4_ufixed60_s$s trajectory_r20wide.py --width-mult 4 --seed $s --widths 38 38 38 77 77 77 154 154 154
done
echo "=== mac_queue_p2l all done $(date) ==="
