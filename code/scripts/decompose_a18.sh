#!/usr/bin/env bash
# Decompose every saved checkpoint of every A18 run (plan 9.6, P9.11: rho_align along
# training) and the final networks of the three arms (plan 9.5, P9.9).
# Usage on the pod, after run_a18_pod.sh:  cd code && PY=python bash scripts/decompose_a18.sh cuda
set -u
PY=${PY:-python}
DEV=${1:-cuda}
OUT=../results/a18
W=$OUT/widths.json
# 9 runs x 17 checkpoints = 153 decompositions.  --positions 16 matches the round-3 CIFAR
# decompositions (P9.12 compares against them); PAR=3 runs the three seeds of an arm in
# parallel with the CPU threads capped (48-vCPU pod, 2026-09-10).
PAR=${PAR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
widths() { $PY -c "import json; print(' '.join(str(w) for w in json.load(open('$W'))['widths']['$1']))"; }
one_run() {  # arm seed
  local arm=$1 s=$2 w run ck ep name
  w=$(widths $arm); run=vgg16_bn_${arm}_s$s
  for ck in $OUT/$run/epoch_*.pt; do
    [ -f "$ck" ] || continue
    ep=$(basename $ck .pt); name=${run}_$ep
    [ -f "$OUT/decompose/${name}_decompose.csv" ] && continue
    $PY scripts/decompose_check.py --cifar-arch vgg16_bn --widths $w --checkpoint $ck \
      --data ../data --device $DEV --out $OUT/decompose --name $name --workers 2 --positions 16 || echo "!! $name failed"
  done
}
for arm in ruler uniform ruler95; do
  if [ "$PAR" -ge 3 ]; then
    for s in 0 1 2; do one_run $arm $s & done; wait
  else
    for s in 0 1 2; do one_run $arm $s; done
  fi
done
echo "=== A18 decompose done $(date) ==="
