#!/usr/bin/env bash
# P2m (notes/controller-criterion-2026-09-16.md, section P2m; hostile review 2026-09-20 route B):
# the width controller on ImageNet-1k ResNet-18, FLOPs-conserving, against a uniform network of the
# same inference MACs and against the A23 full-width runs (results/a18_imagenet_pt, 69.14 / 68.54).
#   1. fetch ImageNet-1k (train + val) into $DATA as 256-short-side JPEG blobs (scripts/fetch_imagenet.py;
#      needs HF_TOKEN of an account that accepted the ILSVRC terms on huggingface.co/datasets/ILSVRC/imagenet-1k)
#   2. pair f60      : controller, budget 0.60 of the full network's forward MACs at 224, acts after
#                      epochs 10 and 25 (grow allowed at 10, shrink only at 25), seeds 0 and 1 in parallel
#   3. pair ufixed60 : uniform block-internal widths x0.567 = MACs 0.599 / params 0.593, trained from scratch
#   4. (APRIORI=1)   : pair apriori60, the controller also acting on the untrained network (epoch 0);
#                      only if P-h1 held on VGG-16 (three seeds), else skipped
# Each run writes result.json when done and resumes itself from resume.pt (at the widths it had) if the
# pod restarts, so re-running this script continues where it stopped.  Two runs share one 4090
# (A23: 14.6-15.1 h per run that way); the whole script is about 31 h + fetch.
# Usage on the pod:
#   cd code && DATA=/workspace/imagenet HF_TOKEN=hf_... nohup bash scripts/run_p2m_imagenet_pod.sh > ../results/p2m.log 2>&1 &
# Watch:  tail -f ../results/p2m.log ../results/p2m/r18_f60_s0.log
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
OUT=${OUT:-../results/p2m}
EXTRA=${EXTRA:-}                                 # dry runs only, e.g. "--epochs 1 --max-train-batches 5 --realloc-epochs 1"
WORKERS=${WORKERS:-$(( $(nproc) / 2 ))}          # loader workers PER RUN; two runs share the CPUs
FETCH_WORKERS=${FETCH_WORKERS:-$(nproc)}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT

if [ ! -f "$DATA/.fetched" ]; then
  echo "=== fetch ImageNet-1k to $DATA $(date) ==="
  $PY scripts/fetch_imagenet.py --out $DATA --split both --workers $FETCH_WORKERS > $OUT/fetch.log 2>&1 \
    || { echo "fetch failed, see $OUT/fetch.log"; exit 1; }
  ntr=$($PY -c "from scripts.imagenet_blob import count_images; print(count_images('$DATA/train'))")
  nva=$($PY -c "from scripts.imagenet_blob import count_images; print(count_images('$DATA/val'))")
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
  for s in 0 1; do run r18_${arm}_s$s --seed $s "$@" & done; wait
}

CTRL="--realloc-budget 0.6 --realloc-cost flops --realloc-calib 6400 --realloc-grow-cap 0.25 --realloc-min-width 8"
pair f60      $CTRL --realloc-epochs 10 25
pair ufixed60 --widths 64 64 36 36 128 73 73 256 145 145 512 290 290
if [ -n "${APRIORI:-}" ]; then pair apriori60 $CTRL --realloc-epochs 0 10 25; fi
echo "=== P2m done $(date) ==="
for f in $OUT/r18_*/result.json; do
  echo -n "$(basename $(dirname $f)): "; $PY -c "import json; r=json.load(open('$f')); print(r['final_top1'], r['final_top5'], r['params_fraction'], r['macs_224'], r['widths'])"
done
