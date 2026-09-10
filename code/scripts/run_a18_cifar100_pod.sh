#!/usr/bin/env bash
# A18 on CIFAR-100 (analysis-plan 9.7): the same four arms as run_a18_pod.sh, on a second
# dataset with the same recipe and a 100-way head.  Full-width arm first (three seeds), then
# the widths are read off the seed-0 full network on 6,400 TRAINING images with
# width_from_ruler.py, then ruler / uniform / ruler95 (three seeds each).  No checkpoints
# (the CIFAR-100 arms are an accuracy test, not a trajectory).
# Usage on the pod:  cd code && PAR=3 nohup bash scripts/run_a18_cifar100_pod.sh > ../results/a18_cifar100.log 2>&1 &
set -u
PY=${PY:-python}
OUT=../results/a18_cifar100
W=$OUT/widths.json
PAR=${PAR:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
COMMON="--device cuda --epochs 100 --workers 2 --gpu-data --data-root ../data --arch vgg16_bn --dataset cifar100"
mkdir -p $OUT

widths() { $PY -c "import json; print(' '.join(str(w) for w in json.load(open('$W'))['widths']['$1']))"; }
run() {  # name, then trajectory_cifar.py args (foreground)
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_100_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_cifar.py $COMMON "$@" --out $OUT/$name > $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
seeds_of() {  # arm; runs the three seeds, in parallel when PAR >= 3
  local arm=$1; shift
  if [ "$PAR" -ge 3 ]; then
    for s in 0 1 2; do run vgg16_bn_${arm}_s$s "$@" --seed $s & done; wait
  else
    for s in 0 1 2; do run vgg16_bn_${arm}_s$s "$@" --seed $s; done
  fi
}

seeds_of full
if [ ! -f "$W" ]; then
  echo "=== widths from the seed-0 full network $(date) ==="
  $PY scripts/width_from_ruler.py --dataset cifar100 --checkpoint $OUT/vgg16_bn_full_s0/vgg16_bn_cifar100_final.pt \
    --data ../data --device cuda --out $W | tee $OUT/widths.log
fi
for arm in ruler uniform ruler95; do
  w=$(widths $arm); echo "arm $arm widths: $w"
  seeds_of $arm --widths $w
done
echo "=== A18 cifar100 done $(date) ==="
