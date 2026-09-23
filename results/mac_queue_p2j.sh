#!/bin/bash
# P2j (2026-09-23, author: "全部做，先把 Mac 免費的項目排進隊列"; hostile review 09-20 S7 and S4):
#   (1) seeds 3-4 for every arm the P2 comparisons use, so each comparison is 5 vs 5 (all-above-all p = 1/C(10,5)
#       = 0.004 under the null instead of 1/C(6,3) = 0.05): full width (round3/a5_vgg16_bn_none), ruler_act, ruler,
#       uniform (A18 arms; seeds 0-2 were trained on the pod with --gpu-data, seeds 3-4 here on MPS, same recipe
#       otherwise -- report the device split), then the controller runs b673, b345 (params) and f88, f60, ffixed342
#       (FLOPs-conserving).
#   (2) sensitivity of the controller's own settings at b673 seed 0 (S4: "four new hyperparameters"): trigger
#       epochs 10/30, 20/50, 15 alone, 15/40/70; calibration images 1,600; minimum width 16. Grow-cap was
#       scanned in P2e.
# 24 runs, ~2 h each. Waits for mac_queue_p2i.sh. Resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2j.sh >> results/mac_queue_p2j.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
until pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e|p2f|p2g|p2h|p2i)\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep (shutdown) counts as busy
until pgrep -f 'scripts/trajectory_cifar.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_queue_p2j start/resume $(date) ==="
UNIFORM="37 37 75 75 149 149 149 298 298 298 298 298 298"
RULER="14 47 107 121 238 247 249 456 376 261 40 40 40"
RULER_ACT="47 64 127 128 255 255 254 504 486 469 347 285 168"
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 --arch vgg16_bn --dataset cifar10 "$@" --out $out
}
# (1) headline comparison first: full, ruler_act, b673
for s in 3 4; do
  run ../results/round3/a5_vgg16_bn_none_s$s --seed $s
  run ../results/a18/vgg16_bn_ruler_act_s$s  --seed $s --widths $RULER_ACT
  run ../results/p2/c10_b673_s$s             --seed $s --realloc-budget 0.673
done
for s in 3 4; do
  run ../results/a18/vgg16_bn_uniform_s$s --seed $s --widths $UNIFORM
  run ../results/a18/vgg16_bn_ruler_s$s   --seed $s --widths $RULER
  run ../results/p2/c10_b345_s$s          --seed $s --realloc-budget 0.345
done
for s in 3 4; do
  run ../results/p2/c10_f88_s$s       --seed $s --realloc-cost flops --realloc-budget 0.88
  run ../results/p2/c10_f60_s$s       --seed $s --realloc-cost flops --realloc-budget 0.60
  run ../results/p2/c10_ffixed342_s$s --seed $s --realloc-cost flops --widths $UNIFORM --realloc-budget 0.342
done
# (2) sensitivity scan, seed 0, budget 0.673 params (reference: c10_b673_s0 94.11 with 15/40, calib 6400, min width 8)
run ../results/p2/c10_b673_ep1030_s0    --seed 0 --realloc-budget 0.673 --realloc-epochs 10 30
run ../results/p2/c10_b673_ep2050_s0    --seed 0 --realloc-budget 0.673 --realloc-epochs 20 50
run ../results/p2/c10_b673_ep15_s0      --seed 0 --realloc-budget 0.673 --realloc-epochs 15
run ../results/p2/c10_b673_ep154070_s0  --seed 0 --realloc-budget 0.673 --realloc-epochs 15 40 70
run ../results/p2/c10_b673_calib1600_s0 --seed 0 --realloc-budget 0.673 --realloc-calib 1600
run ../results/p2/c10_b673_minw16_s0    --seed 0 --realloc-budget 0.673 --realloc-min-width 16
echo "=== mac_queue_p2j all done $(date) ==="
