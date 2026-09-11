#!/bin/bash
# A20 local queue (plan 9.9): arrival now; corrupt + probe once the datasets are down.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
[ -f ../results/repr/arrival.csv ] || $PY scripts/repr_diagnostics.py arrival --results ../results --data ../data --device mps
until grep -q "a20 downloads done" ../results/a20_download.log; do sleep 60; done
$PY scripts/repr_diagnostics.py corrupt --results ../results --data ../data --device mps
$PY scripts/repr_diagnostics.py probe --results ../results --data ../data --device mps
echo "=== a20 local done $(date) ==="
