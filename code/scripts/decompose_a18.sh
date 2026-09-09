#!/usr/bin/env bash
# Decompose every saved checkpoint of every A18 run (plan 9.6, P9.11: rho_align along
# training) and the final networks of the three arms (plan 9.5, P9.9).
# Usage on the pod, after run_a18_pod.sh:  cd code && PY=python bash scripts/decompose_a18.sh cuda
set -u
PY=${PY:-python}
DEV=${1:-cuda}
OUT=../results/a18
W=$OUT/widths.json
widths() { $PY -c "import json; print(' '.join(str(w) for w in json.load(open('$W'))['widths']['$1']))"; }
for arm in ruler uniform ruler95; do
  w=$(widths $arm)
  for s in 0 1 2; do
    run=vgg16_bn_${arm}_s$s
    for ck in $OUT/$run/epoch_*.pt; do
      [ -f "$ck" ] || continue
      ep=$(basename $ck .pt)
      name=${run}_$ep
      [ -f "$OUT/decompose/${name}_decompose.csv" ] && continue
      $PY scripts/decompose_check.py --cifar-arch vgg16_bn --widths $w --checkpoint $ck \
        --data ../data --device $DEV --out $OUT/decompose --name $name --workers 2 || echo "!! $name failed"
    done
  done
done
echo "=== A18 decompose done $(date) ==="
