#!/bin/bash
# A35 (analysis-plan 9.25, 2026-09-30): stationarity, rank-K tail and within-class variance on the 21 CIFAR checkpoints,
# Mac CPU, after the A33 measurement (results/mac_measure_bplus3.sh).  About 40 min per checkpoint under load; resumable.
#   cd <repo> && nohup caffeinate -i results/mac_measure_bplus4.sh >> results/mac_measure_bplus4.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'results/mac_measure_bplus3\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
until pgrep -f 'scripts/measure_init_axis\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_measure_bplus4 start/resume $(date) ==="
$PY scripts/measure_stationarity.py --out ../results/stationarity --device cpu
echo "=== mac_measure_bplus4 all done $(date) ==="
