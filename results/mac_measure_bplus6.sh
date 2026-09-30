#!/bin/bash
# B7 gradient-Gram measurement (theory-chain note B7, 2026-09-30): Sigma_out and Gamma per convolution on the 21 A35
# checkpoints, 3,200 images x 16 positions, Mac CPU about 12 h; after the B2 re-measurement (bplus5).
#   cd <repo> && nohup caffeinate -i results/mac_measure_bplus6.sh >> results/mac_measure_bplus6.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'results/mac_measure_bplus5\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
until pgrep -f 'scripts/measure_residual\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_measure_bplus6 start/resume $(date) ==="
$PY scripts/measure_gradgram.py --out ../results/gradgram --device cpu
$PY scripts/measure_gradgram.py --out ../results/gradgram --judge
echo "=== mac_measure_bplus6 all done $(date) ==="
