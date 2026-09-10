#!/usr/bin/env bash
# Preflight of analysis-plan 9.11 (bar A23): the ResNet-18 arms on CIFAR-10, three seeds each,
# to validate the stage width rule (scripts/resnet_widths.py) and the scripts before ImageNet.
#   (a) full      resnet18 default widths                    -> results/a18_r18/resnet18_full_s{0,1,2}
#   (e) ruler_act widths from width_from_ruler.py --arch resnet18 --tensor act on (a) seed 0,
#                 6,400 TRAINING images                     -> results/a18_r18/resnet18_ruler_act_s{0,1,2}
#   (c) uniform   one factor, same parameter count as (e)   -> results/a18_r18/resnet18_uniform_s{0,1,2}
# Same recipe as every A18 arm (trajectory_cifar.py: OneCycle 100 epochs, batch 128).
# Usage on the pod:  cd code && PAR=3 nohup bash scripts/run_a18_r18_cifar_pod.sh > ../results/a18_r18.log 2>&1 &
# On the Mac (slow, ~2 h per run on MPS): DEVICE=mps PAR=1 bash scripts/run_a18_r18_cifar_pod.sh
set -u
PY=${PY:-python}
PAR=${PAR:-1}
DEVICE=${DEVICE:-cuda}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
OUT=${OUT:-../results/a18_r18}
EPOCHS=${EPOCHS:-100}      # EPOCHS / EXTRA only for dry runs
EXTRA=${EXTRA:-}
mkdir -p $OUT
GPUDATA=$([ "$DEVICE" = cuda ] && echo --gpu-data || echo "")

run() {  # name, then trajectory_cifar.py args
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_$(printf %03d $EPOCHS)_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_cifar.py --device $DEVICE --epochs $EPOCHS --workers 2 $GPUDATA --data-root ../data \
      --arch resnet18 --schedule 0 $EPOCHS $EXTRA "$@" --out $OUT/$name > $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
seeds() {  # arm, then extra args: the three seeds, PAR-way parallel
  local arm=$1; shift
  if [ "$PAR" -ge 3 ]; then
    for s in 0 1 2; do run resnet18_${arm}_s$s --seed $s "$@" & done; wait
  else
    for s in 0 1 2; do run resnet18_${arm}_s$s --seed $s "$@"; done
  fi
}

seeds full
W=$OUT/widths_act.json
if [ ! -f "$W" ]; then
  echo "=== widths from resnet18_full_s0 (training images, after ReLU) $(date) ==="
  $PY scripts/width_from_ruler.py --arch resnet18 --tensor act --dataset cifar10 --device $DEVICE \
      --checkpoint $OUT/resnet18_full_s0/resnet18_cifar10_final.pt --data ../data --out $W > $OUT/widths_act.log 2>&1 \
      || { echo "width_from_ruler failed, see $OUT/widths_act.log"; exit 1; }
fi
widths() { $PY -c "import json; print(' '.join(str(w) for w in json.load(open('$W'))['widths']['$1']))"; }
echo "ruler_act widths: $(widths ruler)"; echo "uniform widths: $(widths uniform)"
seeds ruler_act --widths $(widths ruler)
seeds uniform --widths $(widths uniform)
echo "=== A18-R18-CIFAR done $(date) ==="
