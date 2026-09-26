#!/usr/bin/env bash
# A26 (analysis-plan 9.15, 2026-09-26): the ten-class ImageNet task -- does the idle width of a few-class network
# (VGG-16 on CIFAR-10: a third of the parameters after the ReLU at tau = 0.999) appear at ImageNet resolution in a
# residual network whose last stage is 512 wide?  ResNet-18 from scratch on the ten Imagenette classes
# (Howard 2019: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump,
# golf ball, parachute = ImageNet label indices 0 217 482 491 497 566 569 571 574 701), torchvision reference recipe
# with 300 epochs (lr x0.1 at 100 and 200; ~13k images, 51 steps/epoch, 15k steps: optimised long enough, A25),
# three seeds; then the post-activation ruler on 6,400 of its own training images.  ~1 h per seed on a 4090.
#   cd code && DATA=/workspace/imagenet PY=... nohup bash scripts/run_a26_imagenet10_pod.sh > ../results/a26.log 2>&1 &
set -u
PY=${PY:-python}
DATA=${DATA:-/workspace/imagenet}
OUT=${OUT:-../results/a26}
EXTRA=${EXTRA:-}
WORKERS=${WORKERS:-16}
CLASSES="0 217 482 491 497 566 569 571 574 701"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
mkdir -p $OUT
for s in 0 1 2; do
  name=resnet18_in10_s$s
  if [ -f "$OUT/$name/result.json" ]; then echo "=== $name already done, skipping ==="; else
    echo "=== $name start $(date) ==="
    $PY scripts/train_imagenet_r18.py --data $DATA --workers $WORKERS --epochs 300 --milestones 100 200 \
        --class-ids $CLASSES --seed $s $EXTRA --out $OUT/$name >> $OUT/$name.log 2>&1
    echo "=== $name end $(date) ==="
  fi
  if [ ! -f "$OUT/widths_act_s$s.json" ]; then
    echo "=== ruler $name (6,400 training images of the task, after the ReLU) $(date) ==="
    $PY scripts/width_from_ruler.py --arch resnet18 --tensor act --dataset imagenet --device cuda --workers 8 \
        --checkpoint $OUT/$name/final.pt --data $DATA/train --class-ids $CLASSES --out $OUT/widths_act_s$s.json \
        > $OUT/widths_act_s$s.log 2>&1 || echo "ruler failed for seed $s, see $OUT/widths_act_s$s.log"
  fi
done
echo "=== A26 done $(date) ==="
for s in 0 1 2; do
  f=$OUT/widths_act_s$s.json; [ -f "$f" ] || continue
  $PY - "$f" "$OUT/resnet18_in10_s$s/result.json" <<'PYEOF'
import json, sys
w, r = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
full, ruler = w["widths"]["full"], w["widths"]["ruler"]
frac = [a / b for a, b in zip(ruler, full)]
print(f"seed {r['seed']}: top1 {r['final_top1']:.2f}  params kept {w['params']['ruler'] / w['params']['full']:.3f}  "
      f"ruler {ruler}  stage1 mids {frac[2]:.2f}/{frac[3]:.2f}  stage4 mids {frac[11]:.2f}/{frac[12]:.2f}  stage4 out {frac[10]:.2f}")
PYEOF
done
