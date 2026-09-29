#!/bin/bash
# B+ measurements on the Mac CPU, second list (2026-09-29): waits for mac_measure_bplus.sh, then
#   A32 (plan 9.22): patch gradient outer products against the input covariance      scripts/measure_agop.py
# Skips what is done.  Kept in a second file because the first was already running when this was added.
#   cd <repo> && nohup caffeinate -i results/mac_measure_bplus2.sh >> results/mac_measure_bplus2.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'results/mac_measure_bplus\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
until pgrep -f 'scripts/measure_(a30|controls|residual)\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_measure_bplus2 start/resume $(date) ==="
$PY scripts/measure_agop.py --out ../results/agop --workers 2
echo "=== mac_measure_bplus2 all done $(date) ==="
