#!/usr/bin/env bash
# A28 (analysis-plan 9.17): what in ImageNet reorders ResNet-50's profile.  ResNet-50 under the A17 recipe, one
# factor changed per cell, two seeds each; the base cell (a thousand classes, 224 pixels) is A17's resnet50_s{0,1}.
#   1. the image lists (scripts/a28_lists.py: a hundred classes, 256 held-out images each, 130 images per class)
#   2. the 80-pixel copy of the dataset for the 64-pixel cells (scripts/a28_downsample.py)
#   3. the cells, cheapest first, both seeds of a cell before the next cell:
#        k100_224 (2.2 h a seed)  k1000_n130_224 (2.2 h)  k100_64 (0.5 h)  k1000_64 (5 h)  k100_64_long (4-5 h)
# The 64-pixel times are estimates; the first epoch of k100_64 prints the measured img/s.
# One run at a time: the pod's memory limit (41 GB) does not hold two loaders.  Each run resumes from resume.pt.
# Budget: 28-31 h on a 4090, US$21-24.
# Usage on the pod (needs the A17 data in $DATA and the section-6 subset in $STD):
#   cd code && PY=/workspace/effwidth/venv/bin/python nohup bash scripts/run_a28_pod.sh > ../results/a28.log 2>&1 < /dev/null &
# Copy back: results/a28/lists.json and results/a28/<run>/{*.csv,*.json} (final.pt optional, 100 MB each)
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
DATA64=${DATA64:-/workspace/imagenet80}
STD=${STD:-/workspace/imagenet_val_6400}
OUT=${OUT:-../results/a28}
EXTRA=${EXTRA:-}                                 # dry runs only, e.g. "--epochs 1 --max-train-batches 5 --n-measure 64"
WORKERS=${WORKERS:-8}
SHRINK_WORKERS=${SHRINK_WORKERS:-32}
SEEDS=${SEEDS:-"0 1"}
CELLS=${CELLS:-"k100_224 k1000_n130_224 k100_64 k1000_64 k100_64_long"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT

if [ ! -f "$OUT/lists.json" ]; then
  echo "=== image lists $(date) ==="
  $PY scripts/a28_lists.py --data $DATA --out $OUT/lists.json || { echo "lists failed"; exit 1; }
fi
if [ ! -f "$DATA64/.shrunk" ]; then
  echo "=== 80-pixel copy to $DATA64 $(date) ==="
  $PY scripts/a28_downsample.py --data $DATA --out $DATA64 --workers $SHRINK_WORKERS > $OUT/downsample.log 2>&1 \
    || { echo "downsample failed, see $OUT/downsample.log"; exit 1; }
  ntr=$($PY -c "from scripts.imagenet_blob import count_images; print(count_images('$DATA64/train'))")
  nva=$($PY -c "from scripts.imagenet_blob import count_images; print(count_images('$DATA64/val'))")
  echo "train $ntr val $nva"
  if [ "$ntr" -ne 1281167 ] || [ "$nva" -ne 50000 ]; then echo "image count wrong; not marking shrunk"; exit 1; fi
  touch $DATA64/.shrunk
fi

run() {  # cell seed
  local cell=$1 s=$2 name=$1_s$2 data=$DATA
  case $cell in *_64*) data=$DATA64 ;; esac
  if [ -f "$OUT/$name/a28.json" ] && [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/train_imagenet_a28.py --cell $cell --data $data --lists $OUT/lists.json --standard $STD \
      --workers $WORKERS --seed $s $EXTRA --out $OUT/$name >> $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
for cell in $CELLS; do for s in $SEEDS; do run $cell $s; done; done
echo "=== A28 done $(date) ==="
for f in $OUT/*/result.json; do
  echo -n "$(basename $(dirname $f)): "; $PY -c "import json; r=json.load(open('$f')); print(r['final_top1'], r['final_top5'], round(r['train_hours'],1), 'h')"
done
