#!/usr/bin/env bash
# Arm (e) of A18, analysis-plan 9.8: widths read AFTER the ReLU (the tensor the next layer
# consumes), from results/<sub>/widths_act.json (width_from_ruler.py --tensor act on the seed-0
# full network, 6,400 training images).  Three seeds per dataset, no checkpoints.
# Usage on the pod:  cd code && PAR=3 nohup bash scripts/run_a18_act_pod.sh > ../results/a18_ruler_act.log 2>&1 &
set -u
PY=${PY:-python}
PAR=${PAR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
run() {  # out-dir, log, then trajectory_cifar.py args
  local out=$1 log=$2; shift 2
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out already done, skipping ==="; return; fi
  echo "=== $out start $(date) ==="
  $PY scripts/trajectory_cifar.py --device cuda --epochs 100 --workers 2 --gpu-data --data-root ../data --arch vgg16_bn "$@" --out $out > $log 2>&1
  echo "=== $out end $(date) ==="
}
for ds in cifar10 cifar100; do
  OUT=$([ $ds = cifar10 ] && echo ../results/a18 || echo ../results/a18_cifar100)
  W=$OUT/widths_act.json
  [ -f "$W" ] || { echo "missing $W"; exit 1; }
  w=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$W'))['widths']['ruler']))")
  echo "$ds ruler_act widths: $w"
  if [ "$PAR" -ge 3 ]; then
    for s in 0 1 2; do run $OUT/vgg16_bn_ruler_act_s$s $OUT/vgg16_bn_ruler_act_s$s.log --dataset $ds --widths $w --seed $s & done; wait
  else
    for s in 0 1 2; do run $OUT/vgg16_bn_ruler_act_s$s $OUT/vgg16_bn_ruler_act_s$s.log --dataset $ds --widths $w --seed $s; done
  fi
done
echo "=== A18 ruler_act done $(date) ==="
