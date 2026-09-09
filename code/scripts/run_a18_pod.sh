#!/usr/bin/env bash
# A18, analysis-plan 9.5: width from the ruler, retrained.  CIFAR-10 VGG-16_BN,
# three arms x three seeds, widths read from results/a18/widths.json (produced by
# scripts/width_from_ruler.py on the seed-0 full-width network, training images).
# Arm (a) full width already exists as results/round3/a5_vgg16_bn_none_s{0,1,2}.
# Every run saves a checkpoint at each measured epoch (plan 9.6, P9.11).
# Usage on the pod:  cd code && nohup bash scripts/run_a18_pod.sh > ../results/a18.log 2>&1 &
set -u
PY=${PY:-python}
OUT=../results/a18
W=$OUT/widths.json
[ -f "$W" ] || { echo "missing $W: run scripts/width_from_ruler.py first"; exit 1; }
COMMON="--device cuda --epochs 100 --workers 2 --gpu-data --data-root ../data --arch vgg16_bn --save-checkpoints"

widths() { $PY -c "import json,sys; print(' '.join(str(w) for w in json.load(open('$W'))['widths']['$1']))"; }

run() {  # name, then trajectory_cifar.py args   (foreground)
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_100_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_cifar.py $COMMON "$@" --out $OUT/$name > $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}

mkdir -p $OUT
for arm in ruler uniform ruler95; do
  w=$(widths $arm)
  echo "arm $arm widths: $w"
  for s in 0 1 2; do run vgg16_bn_${arm}_s$s --widths $w --seed $s; done
done
echo "=== A18 done $(date) ==="
