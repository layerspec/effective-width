#!/usr/bin/env bash
# A27 (analysis-plan 9.16, 2026-09-27): VGG-16 (BN, ImageNet head) on the same ten Imagenette classes and the same
# recipe as A26 (300 epochs, lr x0.1 at 100 and 200), three seeds, then the post-activation ruler on 6,400 of its own
# training images (the VGG 13-layer rule).  Does a plain network collapse its late layers on a few-class task at
# ImageNet resolution where the residual ResNet-18 (A26) did not?  ~1.2 h per seed on a 4090.
#   cd code && DATA=/workspace/imagenet PY=... nohup bash scripts/run_a27_vgg_imagenet10_pod.sh > ../results/a27.log 2>&1 &
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
OUT=${OUT:-../results/a27}
EXTRA=${EXTRA:-}
WORKERS=${WORKERS:-8}
CLASSES="0 217 482 491 497 566 569 571 574 701"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT
for s in 0 1 2; do
  name=vgg16_bn_in10_s$s
  if [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; else
    echo "=== $name start $(date) ==="
    $PY scripts/train_imagenet_r18.py --arch vgg16_bn --data $DATA --workers $WORKERS --epochs 300 --milestones 100 200 \
        --class-ids $CLASSES --seed $s $EXTRA --out $OUT/$name >> $OUT/$name.log 2>&1
    echo "=== $name end $(date) ==="
  fi
  if [ ! -f "$OUT/widths_act_s$s.json" ]; then
    echo "=== ruler $name (6,400 training images of the task, after the ReLU) $(date) ==="
    $PY scripts/width_from_ruler.py --arch vgg16_bn --tensor act --dataset imagenet --device cuda --workers 8 \
        --checkpoint $OUT/$name/final.pt --data $DATA/train --class-ids $CLASSES --out $OUT/widths_act_s$s.json \
        > $OUT/widths_act_s$s.log 2>&1 || echo "ruler failed for seed $s, see $OUT/widths_act_s$s.log"
  fi
done
echo "=== A27 done $(date) ==="
for s in 0 1 2; do
  f=$OUT/widths_act_s$s.json; [ -f "$f" ] || continue
  $PY - "$f" "$OUT/vgg16_bn_in10_s$s/result.json" <<'PYEOF'
import json, sys
w, r = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ratio = [l["k_star_0.999"] / l["C"] for l in w["layers"]]
print(f"seed {r['seed']}: top1 {r['final_top1']:.2f}  params kept {w['params']['ruler'] / w['params']['full']:.3f}  "
      f"first5 {[round(x, 2) for x in ratio[:5]]}  last3 {[round(x, 2) for x in ratio[-3:]]}")
PYEOF
done
