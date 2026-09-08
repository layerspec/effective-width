#!/usr/bin/env bash
# Round 3 on a rented GPU (submission bar A2, A5, A6, A9).  Written 2026-09-07.
# Everything here is pre-registered in notes/analysis-plan.md section 9.
#
#   bash scripts/run_round3_pod.sh [/data/imagenet_val]      # imagenet dir optional (A6)
#   SUBSET6400=/data/imagenet_val_6400 adds the A7 bootstrap; SKIP_TRAJ=1 skips A9
#
# Budget on one 3090 (measured 2026-09-08: ~20 s/epoch for basic52, three seeds in
# parallel): A2 + A5 ~ 5-6 h wall-clock; A6 ~2 h; A7 ~1 h.
# Results land in ../results/round3/ (and results/fullval via run_local_seeds);
# tar them up and copy back (see notes/running-guide.md).  Re-running skips
# finished runs, so the script can be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=../results/round3
mkdir -p $OUT
COMMON="--device cuda --epochs 100 --workers 4 --data-root ../data"

run() {  # name, then trajectory_cifar.py args   (foreground)
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_100_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_cifar.py $COMMON "$@" --out $OUT/$name > $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
# Three seeds of one configuration in parallel: a CIFAR run uses ~2 GB and a
# fraction of the GPU, so three at once roughly halves the wall-clock.
run3() {  # name-prefix, then args (without --seed)
  local prefix=$1; shift
  for s in 0 1 2; do run ${prefix}_s$s "$@" --seed $s & done
  wait
}

# A9: trajectories.  These are being run on the Mac over several days
# (results/cifar_local.sh, resumable); set SKIP_TRAJ=1 to leave them out here.
if [ -z "${SKIP_TRAJ:-}" ]; then
  run trajectory_resnet50_s0 --arch resnet50 --seed 0 &
  run trajectory_vgg16_bn_s1 --arch vgg16_bn --seed 1 &
  run trajectory_vgg16_bn_s2 --arch vgg16_bn --seed 2 &
  wait
fi

# A2: controlled block type, same recipe, same conv count (52 vs 53), 3 seeds each
run3 a2_bottleneck  --arch resnet50
run3 a2_basic52     --arch resnet_basic52
run3 a2_mobilenetv2 --arch mobilenetv2

# A5: orthogonality intervention (the 'none' arms are the a2 / VGG runs; ONI not implemented)
run3 a5_vgg16_bn_none --arch vgg16_bn
run3 a5_vgg16_bn_so   --arch vgg16_bn --ortho so
run3 a5_vgg16_bn_srip --arch vgg16_bn --ortho srip
run3 a5_basic52_so    --arch resnet_basic52 --ortho so
run3 a5_basic52_srip  --arch resnet_basic52 --ortho srip

# A6: the subset measurements repeated on the full validation set (needs the 50k images)
if [ -n "${1:-}" ]; then
  echo "=== A6 full-validation rerun $(date) ==="
  OUT6=../results/fullval; mkdir -p $OUT6
  C6="--data $1 --device cuda --batch-size 64 --workers 8 --positions 16"
  for s in 1 2 3 4 5; do $PY -m layerspec.run $C6 --seed $s --models resnet50_random --out $OUT6/seed$s; done
  $PY -m layerspec.run $C6 --seed 0 --out $OUT6/trained --models resnet50 timm:resnet50.tv2_in1k timm:resnet50.a1_in1k timm:resnet50.a3_in1k timm:resnet50.gluon_in1k timm:resnet50.fb_ssl_yfcc100m_ft_in1k
fi
# A7: bootstrap over image subsets (analysis-plan 9.3).  20 draws of 3,200 images
# from the same 6,400-image subset dir (fetch it with fetch_imagenet_val.py --limit 6400
# --seed 0, or copy data/imagenet_val_6400 from the Mac), six ResNet-50 recipes each.
if [ -n "${SUBSET6400:-}" ]; then
  echo "=== A7 bootstrap $(date) ==="
  OUT7=../results/bootstrap; mkdir -p $OUT7
  for b in $(seq 1 20); do
    $PY -m layerspec.run --data $SUBSET6400 --device cuda --batch-size 64 --workers 8 --positions 16 \
      --limit 3200 --seed $b --out $OUT7/draw$b --models \
      resnet50 timm:resnet50.tv2_in1k timm:resnet50.a1_in1k timm:resnet50.a3_in1k timm:resnet50.gluon_in1k timm:resnet50.fb_ssl_yfcc100m_ft_in1k
  done
fi
echo "=== round 3 done $(date) ==="
