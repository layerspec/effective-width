#!/bin/bash
# A34 part b on MPS (plan 9.24).  The p2l training that holds the GPU is paused with SIGSTOP for the duration and
# resumed with SIGCONT on exit, whatever happens (2026-09-26 precedent).  Usage:
#   results/two_rulers/run_part_b.sh <out dir> [max step]
cd "$(dirname "$0")/../../code" || exit 1
OUT=${1:-../results/two_rulers}; MAX=${2:-1950}
PID=$(pgrep -f 'trajectory_(r20w[i]de|c[i]far|v[i]t)' | head -1)
resume() { [ -n "$PID" ] && kill -CONT $PID 2>/dev/null && echo "=== training $PID resumed $(date) ==="; }
trap resume EXIT
if [ -n "$PID" ]; then kill -STOP $PID && echo "=== training $PID paused $(date) ==="; sleep 5; fi
OMP_NUM_THREADS=4 $HOME/.venvs/effwidth/bin/python scripts/two_rulers.py --part b --out $OUT --device mps --max-step $MAX
echo "=== part b returned $? $(date) ==="
