#!/usr/bin/env bash
# D5 (hostile review 2026-09-15, analysis-plan 9.2 addendum of 2026-09-24): the ONI arm of the
# orthogonality intervention -- every dense kernel an exact isometry by construction (scripts/oni.py,
# T = 20), same recipe, same seeds and same images as the SO / SRIP / none arms of round 3, then the
# same decomposition (out / kernel / data / ortho per layer) so checkpoint_analysis.py section 13
# pairs it with the no-penalty arm.  Six training runs (VGG-16_BN and basic [6,6,6,6], seeds 0-2,
# three seeds in parallel), about 1.5-2 h on one 4090.
#   cd code && nohup bash scripts/run_d5_oni_pod.sh > ../results/d5_oni.log 2>&1 &
# Copy back: results/round3/a5_*_oni_s*/ (the baked *_oni_cifar10_final.pt are git-ignored; keep them
# with the other round-3 weights) and results/round3_decompose/a5_*_oni_s*_decompose.csv.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
OUT=../results/round3; DEC=../results/round3_decompose
mkdir -p $OUT $DEC
COMMON="--device cuda --epochs 100 --workers 2 --gpu-data --data-root ../data"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}   # a 128-core pod thrashes the CPU-side covariance eigendecompositions (2026-09-26)
ITERS=${ONI_ITERS:-20}

run() {  # name, then trajectory_oni.py args   (foreground)
  local name=$1; shift
  if [ -f "$OUT/$name/epoch_100_layers.csv" ]; then echo "=== $name already done, skipping ==="; return; fi
  echo "=== $name start $(date) ==="
  $PY scripts/trajectory_oni.py --oni-iters $ITERS $COMMON "$@" --out $OUT/$name > $OUT/$name.log 2>&1
  echo "=== $name end $(date) ==="
}
run3() {  # name-prefix, then args (without --seed)
  local prefix=$1; shift
  for s in 0 1 2; do run ${prefix}_s$s "$@" --seed $s & done
  wait
}
run3 a5_vgg16_bn_oni --arch vgg16_bn
run3 a5_basic52_oni  --arch resnet_basic52

for arch in vgg16_bn resnet_basic52; do
  short=${arch/resnet_/}
  for s in 0 1 2; do
    name=a5_${short}_oni_s$s; pt=$OUT/$name/${arch}_oni_cifar10_final.pt
    [ -f "$pt" ] || { echo "missing $pt"; continue; }
    [ -f "$DEC/${name}_decompose.csv" ] && continue
    echo "=== decompose $name $(date) ==="
    $PY scripts/decompose_check.py --cifar-arch $arch --checkpoint "$pt" --name "$name" --data ../data \
        --device cuda --workers 2 --positions 16 --out $DEC
  done
done
echo "=== D5 done $(date) ==="
