#!/usr/bin/env bash
# Random-init controls with several seeds, plus the trained ResNet-50 recipes
# on the SAME image subset, run locally on Apple silicon (MPS) at no cost.
#
#   bash scripts/run_local_seeds.sh ../data/imagenet_val_6400 [python]
#
# 6,400 images x 16 positions = 102,400 samples per layer, so n/C >= 50 holds
# up to C = 2048 for the position-sampled statistic.  The pooled statistic
# (n = images) passes the gate only up to C = 128 here; do not report pooled
# rows from this run.  Written 2026-09-04 for the architectural-vs-learned
# question (notes/round2-2026-09-04.md section 6).
set -euo pipefail
DATA=${1:?usage: run_local_seeds.sh <image dir> [python]}
PY=${2:-python}
OUT=../results/local6400
SEEDS=${SEEDS:-"1 2 3 4 5"}
DEVICE=${DEVICE:-mps}
COMMON="--data $DATA --device $DEVICE --batch-size 32 --workers 4 --positions 16"

for s in $SEEDS; do
  echo "=== resnet50_random seed $s ===" >&2
  $PY -m layerspec.run $COMMON --seed $s --models resnet50_random --out $OUT/seed$s
done

echo "=== trained recipes on the same subset ===" >&2
$PY -m layerspec.run $COMMON --seed 0 --out $OUT/trained --models \
  resnet50 timm:resnet50.tv2_in1k timm:resnet50.a1_in1k timm:resnet50.a3_in1k \
  timm:resnet50.gluon_in1k timm:resnet50.fb_ssl_yfcc100m_ft_in1k
echo "=== done ===" >&2
