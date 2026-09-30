#!/usr/bin/env bash
# A29b (analysis-plan 9.18, P9.50; author 2026-09-30: do it): the depth axis under ONE recipe and ONE
# initialisation.  ResNet-18 (20 conv) and ResNet-34 (36) trained with the A17 recipe (SGD, lr 0.1 cosine, 90
# epochs, wd 1e-4, torchvision initialisation), two seeds each, profiles along the trajectory on the section-6
# subset -- exactly the A17 runs of basic [6,6,6,6] (52) and ResNet-50, so the three basic depths are comparable.
# Order: resnet18_s0, resnet34_s0, resnet18_s1, resnet34_s1 (a partial budget still gives one full depth axis).
# ~38-42 h on a 4090.  Each run resumes from resume.pt; re-running continues.
#   cd code && PY=... DATA=/workspace/imagenet MEASURE=/workspace/imagenet_val_6400 nohup bash scripts/run_a29b_pod.sh > ../results/a29b.log 2>&1 &
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
MEASURE=${MEASURE:-/workspace/imagenet_val_6400}
OUT=${OUT:-../results/a29b}
EXTRA=${EXTRA:-}
WORKERS=${WORKERS:-8}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT
[ -f "$DATA/.fetched" ] || { echo "ImageNet not fetched at $DATA"; exit 1; }
[ -f "$MEASURE/.fetched" ] || { echo "measurement subset not fetched at $MEASURE"; exit 1; }

run() {  # name, then train_imagenet_a17.py args
  local name=$1; shift
  if [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/train_imagenet_a17.py --data $DATA --measure-data $MEASURE --workers $WORKERS $EXTRA "$@" \
      --out $OUT/$name >> $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
for s in 0 1; do
  for arch in resnet18 resnet34; do run ${arch}_s$s --arch $arch --seed $s; done
done
echo "=== A29b done $(date) ==="
for f in $OUT/*/result.json; do
  $PY - "$f" <<'PYEOF'
import json, sys, os
r = json.load(open(sys.argv[1]))
print(f"{os.path.basename(os.path.dirname(sys.argv[1]))}: {r['final_top1']:.3f} {r['final_top5']:.3f} {r.get('train_hours', 0):.1f} h")
PYEOF
done
