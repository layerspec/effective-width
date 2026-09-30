#!/bin/bash
# B2 cross-term re-measurement (analysis-plan 9.21 addendum, 2026-09-30): the same six weights and images as results/residual,
# with the cross-covariance singular values stored, into results/residual_b2.  Mac CPU, about 2 h; after A35 (bplus4).
#   cd <repo> && nohup caffeinate -i results/mac_measure_bplus5.sh >> results/mac_measure_bplus5.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
until pgrep -f 'results/mac_measure_bplus4\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep counts as busy
until pgrep -f 'scripts/measure_stationarity\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_measure_bplus5 start/resume $(date) ==="
$PY scripts/measure_residual.py --out ../results/residual_b2 --device cpu
$PY scripts/measure_residual.py --out ../results/residual_b2 --judge
echo "=== mac_measure_bplus5 all done $(date) ==="
