#!/usr/bin/env bash
# Analysis-plan 9.11 (bar A23): ResNet-18 on ImageNet-1k from scratch, torchvision reference
# recipe, three arms x two seeds, run in the order the dependencies force:
#   1. fetch ImageNet-1k (train + val) into $DATA as 256-short-side JPEGs (scripts/fetch_imagenet.py)
#   2. (a) full width, seeds 0 and 1               (two runs in parallel on one GPU)
#   3. widths_act.json from (a) seed 0 on 6,400 TRAINING images, after the ReLU, stage rule
#   4. (e) ruler_act, seeds 0 and 1
#   5. (c) uniform (same parameter count as (e)), seeds 0 and 1
# Each run writes result.json when done and resumes itself from resume.pt if the pod restarts,
# so re-running this script continues where it stopped.
# Usage on the pod:
#   cd code && DATA=/workspace/imagenet nohup bash scripts/run_a18_imagenet_pod.sh > ../results/a18_imagenet.log 2>&1 &
# Watch:  tail -f ../results/a18_imagenet.log ../results/a18_imagenet/resnet18_full_s0.log
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
OUT=${OUT:-../results/a18_imagenet}
EXTRA=${EXTRA:-}                                 # dry runs only, e.g. "--epochs 1 --max-train-batches 5"
WORKERS=${WORKERS:-$(( $(nproc) / 2 ))}          # loader workers PER RUN; two runs share the CPUs
FETCH_WORKERS=${FETCH_WORKERS:-$(nproc)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT

if [ ! -f "$DATA/.fetched" ]; then
  echo "=== fetch ImageNet-1k to $DATA $(date) ==="
  $PY scripts/fetch_imagenet.py --out $DATA --split both --workers $FETCH_WORKERS > $OUT/fetch.log 2>&1 \
    || { echo "fetch failed, see $OUT/fetch.log"; exit 1; }
  ntr=$(find $DATA/train -name '*.JPEG' | wc -l); nva=$(find $DATA/val -name '*.JPEG' | wc -l)
  echo "train $ntr val $nva"
  if [ "$ntr" -ne 1281167 ] || [ "$nva" -ne 50000 ]; then echo "image count wrong; not marking fetched"; exit 1; fi
  touch $DATA/.fetched
  rm -rf $DATA/.parquet
fi

run() {  # name, then train_imagenet_r18.py args
  local name=$1; shift
  if [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/train_imagenet_r18.py --data $DATA --workers $WORKERS $EXTRA "$@" --out $OUT/$name >> $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
pair() {  # arm, then extra args: seeds 0 and 1 in parallel
  local arm=$1; shift
  for s in 0 1; do run resnet18_${arm}_s$s --seed $s "$@" & done; wait
}

pair full
W=$OUT/widths_act.json
if [ ! -f "$W" ]; then
  echo "=== widths from resnet18_full_s0 (6,400 training images, after ReLU, stage rule) $(date) ==="
  $PY scripts/width_from_ruler.py --arch resnet18 --tensor act --dataset imagenet --device cuda --workers 8 \
      --checkpoint $OUT/resnet18_full_s0/final.pt --data $DATA/train --out $W > $OUT/widths_act.log 2>&1 \
      || { echo "width_from_ruler failed, see $OUT/widths_act.log"; exit 1; }
fi
widths() { $PY -c "import json; print(' '.join(str(w) for w in json.load(open('$W'))['widths']['$1']))"; }
echo "ruler_act widths: $(widths ruler)"; echo "uniform widths: $(widths uniform)"
pair ruler_act --widths $(widths ruler)
pair uniform --widths $(widths uniform)
echo "=== A18-ImageNet done $(date) ==="
