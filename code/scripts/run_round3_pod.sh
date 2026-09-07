#!/usr/bin/env bash
# Round 3 on a rented GPU (submission bar A2, A5, A6, A9).  Written 2026-09-07.
# Everything here is pre-registered in notes/analysis-plan.md section 9.
#
#   bash scripts/run_round3_pod.sh [/data/imagenet_val]      # imagenet dir optional (A6)
#   SUBSET6400=/data/imagenet_val_6400 adds the A7 bootstrap; SKIP_TRAJ=1 skips A9
#
# Budget on one 3090: CIFAR runs ~10-25 min each x 33 runs ~ 6-9 h; A6 ~2 h.
# Results land in ../results/round3/ (and results/fullval via run_local_seeds);
# tar them up and copy back (see notes/running-guide.md).  Re-running skips
# finished runs, so the script can be restarted after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=../results/round3
mkdir -p $OUT
COMMON="--device cuda --epochs 100 --workers 8 --data-root ../data"

run() {  # name, then trajectory_cifar.py args
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_100_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_cifar.py $COMMON "$@" --out $OUT/$name
}

# A9: trajectories.  These are being run on the Mac over several days
# (results/cifar_local.sh, resumable); set SKIP_TRAJ=1 to leave them out here.
if [ -z "${SKIP_TRAJ:-}" ]; then
  run trajectory_resnet50_s0   --arch resnet50 --seed 0
  run trajectory_vgg16_bn_s1   --arch vgg16_bn --seed 1
  run trajectory_vgg16_bn_s2   --arch vgg16_bn --seed 2
fi

# A2: controlled block type, same recipe, same conv count (52 vs 53), 3 seeds each
# (trajectory_resnet50_s0 doubles as a2_bottleneck_s0)
for s in 0 1 2; do
  run a2_basic52_s$s      --arch resnet_basic52 --seed $s
  if [ $s -gt 0 ] || [ -n "${SKIP_TRAJ:-}" ]; then run a2_bottleneck_s$s --arch resnet50 --seed $s; fi
  run a2_mobilenetv2_s$s  --arch mobilenetv2 --seed $s
done

# A5: orthogonality intervention, three arms x two archs x 3 seeds
# (the 'none' arms are the trajectory / a2 runs above; ONI not implemented)
for s in 0 1 2; do
  for o in so srip; do
    run a5_vgg16_bn_${o}_s$s   --arch vgg16_bn --ortho $o --seed $s
    run a5_basic52_${o}_s$s    --arch resnet_basic52 --ortho $o --seed $s
  done
done
# the VGG 'none' arms for seeds 1,2 are trajectory_vgg16_bn_s1/s2; seed 0 is results/trajectory

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
