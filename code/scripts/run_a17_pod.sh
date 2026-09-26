#!/usr/bin/env bash
# A17 reduced (submission bar D2; analysis-plan 9.4 + 2026-09-25 addendum): basic [6,6,6,6] vs
# bottleneck ResNet-50 on ImageNet-1k, ONE recipe (SGD, lr 0.1 cosine, 90 epochs, wd 1e-4), two seeds
# each, profiles measured along the trajectory on the same 6,400 validation images as the public
# checkpoints (section 6 protocol).  Order:
#   1. fetch ImageNet-1k train + val as 256-short-side JPEG blobs (scripts/fetch_imagenet.py; needs HF_TOKEN
#      of an account that accepted the ILSVRC terms) -- 8 min on the A23 pod
#   2. the 6,400-image measurement subset: scripts/fetch_imagenet_val.py --limit 6400 (sorted parquet order,
#      the same 6,400 files as data/imagenet_val_6400 on the Mac; or rsync that folder, 647 MB)
#   3. pair 0: resnet50_s0 then basic6666_s0 (sequential; about 50 h, US$38 on a 4090)
#   4. pair 1: the same with seed 1 (SEED0_ONLY=1 skips it; re-run the script later to add it)
# Each run resumes itself from resume.pt if the pod restarts; re-running this script continues.
# Budget: about 100 GPU-hours on a 4090 (US$75-95), less wall-clock on an A100/H100 at a similar price.
# Usage on the pod:
#   cd code && DATA=/workspace/imagenet HF_TOKEN=hf_... nohup bash scripts/run_a17_pod.sh > ../results/a17.log 2>&1 &
# Watch:  tail -f ../results/a17.log ../results/a17/resnet50_s0.log
# Copy back: results/a17/<run>/{epochs.csv,result.json,trajectory_layers.csv,epoch_*_layers.csv} (final.pt optional, 100 MB each)
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
MEASURE=${MEASURE:-/workspace/imagenet_val_6400}
OUT=${OUT:-../results/a17}
EXTRA=${EXTRA:-}                                 # dry runs only, e.g. "--epochs 1 --max-train-batches 5 --measure-limit 64"
WORKERS=${WORKERS:-16}                           # loader workers per run (16 is plenty; nproc/2 on a 128-core pod is not)
FETCH_WORKERS=${FETCH_WORKERS:-16}                # each worker holds a ~1 GB parquet shard on disk
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
if [ ! -f "$MEASURE/.fetched" ]; then
  echo "=== fetch the 6,400-image measurement subset to $MEASURE $(date) ==="
  $PY scripts/fetch_imagenet_val.py --out $MEASURE --limit 6400 > $OUT/fetch_val6400.log 2>&1 \
    || { echo "subset fetch failed, see $OUT/fetch_val6400.log"; exit 1; }
  n=$(ls $MEASURE | grep -c -i 'jpeg$\|jpg$')
  echo "measurement images: $n"
  if [ "$n" -ne 6400 ]; then echo "expected 6400 images; not marking fetched"; exit 1; fi
  touch $MEASURE/.fetched
fi

run() {  # name, then train_imagenet_a17.py args
  local name=$1; shift
  if [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/train_imagenet_a17.py --data $DATA --measure-data $MEASURE --workers $WORKERS $EXTRA "$@" \
      --out $OUT/$name >> $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
pair() {  # seed: both architectures, one seed of each first so a partial budget still gives a full comparison.
  # Sequential by default: ResNet-50 and basic [6,6,6,6] at batch 256 need 10-13 GB each, which does not fit a
  # 24 GB card together with anything else, and both are GPU-bound so parallel would not be faster. PARALLEL=1 overrides.
  local s=$1; shift
  if [ -n "${PARALLEL:-}" ]; then
    for arch in resnet50 basic6666; do run ${arch}_s$s --arch $arch --seed $s "$@" & done; wait
  else
    for arch in resnet50 basic6666; do run ${arch}_s$s --arch $arch --seed $s "$@"; done
  fi
}

pair 0
if [ -z "${SEED0_ONLY:-}" ]; then pair 1; fi
echo "=== A17 done $(date) ==="
for f in $OUT/*/result.json; do
  echo -n "$(basename $(dirname $f)): "; $PY -c "import json; r=json.load(open('$f')); print(r['final_top1'], r['final_top5'], round(r['train_hours'],1), 'h')"
done
