#!/bin/bash
# B+ measurements on the Mac CPU (2026-09-29), one at a time, after the A30 transformer measurement:
#   1. controls on the measurement (plan 9.20): five weights with their own input statistics, then the
#      gamma = 1 random networks of the five ResNet families        scripts/measure_controls.py
#   2. the residual stream at the addition (plan 9.21, Th9)          scripts/measure_residual.py, if present
# Every step skips what is done, so re-running this script continues.  Training queues use the GPU (MPS) and
# are not waited for.
#   cd <repo> && nohup caffeinate -i results/mac_measure_bplus.sh >> results/mac_measure_bplus.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'scripts/measure_a30\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
echo "=== mac_measure_bplus start/resume $(date) ==="
$PY scripts/measure_a30.py --out ../results/a30 --workers 2          # finishes A30 if it was interrupted
$PY scripts/measure_controls.py --out ../results/bplus_controls --workers 2
if [ -f scripts/measure_residual.py ]; then
  $PY scripts/measure_residual.py --out ../results/residual --workers 2
fi
echo "=== mac_measure_bplus all done $(date) ==="
