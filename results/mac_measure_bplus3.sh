#!/bin/bash
# A33 (analysis-plan 9.23, 2026-09-30): the missing reference networks of each initialisation, on the Mac CPU, one
# at a time, after the A32 measurement (results/mac_measure_bplus2.sh).  15 runs, 7-8 h.  Resumable.
#   cd <repo> && nohup caffeinate -i results/mac_measure_bplus3.sh >> results/mac_measure_bplus3.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'results/mac_measure_bplus2?\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
until pgrep -f 'scripts/measure_(a30|controls|residual|agop)\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_measure_bplus3 start/resume $(date) ==="
$PY scripts/measure_init_axis.py --out ../results/init_axis --workers 2
echo "=== mac_measure_bplus3 all done $(date) ==="
