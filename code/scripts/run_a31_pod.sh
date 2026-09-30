#!/usr/bin/env bash
# A31 (analysis-plan 9.26): the class count alone on ImageNet images at 224 pixels.  ResNet-50 under the A17
# recipe, 1,000 training images per class, 37,000 steps per cell, K = 10 / 20 / 50 / 100, two seeds each.
#   1. the lists (scripts/a31_lists.py: 100 classes with >= 1,256 images, nested order, 256 held-out + 1,000 training each)
#   2. the cells: the two ends first (k10, k100), then k20, k50; both seeds of a cell before the next one.
# Every run is ~1.6-1.8 h on a 4090 (37,000 steps at ~1,700 img/s plus measurement); 8 runs ~14 h, US$11.
# One run at a time (41 GB memory limit).  Each run resumes from resume.pt.
#   cd code && PY=/workspace/effwidth/venv/bin/python nohup bash scripts/run_a31_pod.sh > ../results/a31.log 2>&1 < /dev/null &
# Copy back: results/a31/lists.json and results/a31/<run>/{*.csv,*.json} (final.pt optional)
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
STD=${STD:-/workspace/imagenet_val_6400}
OUT=${OUT:-../results/a31}
EXTRA=${EXTRA:-}                                 # dry runs only, e.g. "--epochs 1 --max-train-batches 5 --n-measure 64"
WORKERS=${WORKERS:-8}
SEEDS=${SEEDS:-"0 1"}
CELLS=${CELLS:-"k10 k100 k20 k50"}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT
[ -f "$DATA/.fetched" ] || { echo "ImageNet not fetched at $DATA"; exit 1; }
[ -f "$STD/.fetched" ] || { echo "measurement subset not fetched at $STD"; exit 1; }

if [ ! -f "$OUT/lists.json" ]; then
  echo "=== lists $(date) ==="
  $PY scripts/a31_lists.py --data $DATA --out $OUT/lists.json || { echo "lists failed"; exit 1; }
fi

run() {  # cell seed
  local cell=$1 s=$2 name=$1_s$2
  if [ -f "$OUT/$name/a31.json" ] && [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/train_imagenet_a31.py --cell $cell --data $DATA --lists $OUT/lists.json --standard $STD \
      --workers $WORKERS --seed $s $EXTRA --out $OUT/$name >> $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
for cell in $CELLS; do for s in $SEEDS; do run $cell $s; done; done
echo "=== A31 done $(date) ==="
for f in $OUT/*/result.json; do
  echo -n "$(basename $(dirname $f)): "; $PY -c "import json; r=json.load(open('$f')); print(r['final_top1'], r['final_top5'], round(r['train_hours'],1), 'h')"
done
