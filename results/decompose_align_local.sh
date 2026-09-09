#!/usr/bin/env bash
# Re-decompose the 24 round-3 final networks with rho_align (plan 9.6, P9.12), after the
# CIFAR trajectory queue is done (MPS contention).  Resumable: finished CSVs are skipped.
cd "$(dirname "$0")/../code"
while pgrep -f 'results/cifar_local.sh' > /dev/null; do sleep 300; done
echo "=== decompose_align_local start $(date) ==="
PT_ROOT=../results/round3_pt bash scripts/decompose_round3.sh mps
echo "=== decompose_align_local done $(date) ==="
