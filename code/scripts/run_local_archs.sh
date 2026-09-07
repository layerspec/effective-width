#!/usr/bin/env bash
# Cross-architecture version of run_local_seeds.sh: several trained
# checkpoints per architecture plus three random-init seeds each, on the same
# 6,400-image subset, on the Mac (MPS).  For the architectural-vs-learned
# question (notes/round2-2026-09-04.md section 6): is "the profile is learned"
# a ResNet-50 fact or a general one?
#
#   bash scripts/run_local_archs.sh ../data/imagenet_val_6400 [python]
#
# Checkpoint names verified against timm.list_pretrained on 2026-09-04.
# MobileNetV2 has a single ImageNet-1k checkpoint on timm and is left out.
set -euo pipefail
DATA=${1:?usage: run_local_archs.sh <image dir> [python]}
PY=${2:-python}
OUT=../results/local6400/archs
DEVICE=${DEVICE:-mps}
COMMON="--data $DATA --device $DEVICE --batch-size 32 --workers 4 --positions 16"

TRAINED="
timm:resnet18.tv_in1k timm:resnet18.a1_in1k timm:resnet18.a2_in1k timm:resnet18.a3_in1k timm:resnet18.gluon_in1k timm:resnet18.fb_ssl_yfcc100m_ft_in1k
timm:resnet34.tv_in1k timm:resnet34.a1_in1k timm:resnet34.a2_in1k timm:resnet34.a3_in1k timm:resnet34.gluon_in1k
timm:vgg16.tv_in1k timm:vgg16_bn.tv_in1k
timm:densenet121.tv_in1k timm:densenet121.ra_in1k
timm:mobilenetv3_large_100.ra_in1k timm:mobilenetv3_large_100.ra4_e3600_r224_in1k timm:mobilenetv3_large_100.miil_in21k_ft_in1k
timm:efficientnet_b0.ra_in1k timm:efficientnet_b0.ra4_e3600_r224_in1k
timm:tf_efficientnet_b0.in1k timm:tf_efficientnet_b0.aa_in1k timm:tf_efficientnet_b0.ap_in1k timm:tf_mobilenetv3_large_100.in1k
timm:resnet101.tv_in1k timm:resnet101.tv2_in1k timm:resnet101.a1_in1k timm:resnet101.a3_in1k timm:resnet101.gluon_in1k
timm:wide_resnet50_2.tv_in1k timm:wide_resnet50_2.tv2_in1k timm:wide_resnet50_2.racm_in1k
timm:resnext50_32x4d.tv_in1k timm:resnext50_32x4d.tv2_in1k timm:resnext50_32x4d.a1_in1k timm:resnext50_32x4d.a3_in1k timm:resnext50_32x4d.gluon_in1k
timm:mobilenetv2_100.ra_in1k
"
# 2026-09-07: the other bottleneck architectures (depth 101; width 2x; grouped 3x3)
# and MobileNetV2, so that "bottleneck" is four families and not one.
# The four tf_* checkpoints (added 2026-09-07) are the same architectures with
# TensorFlow 'SAME' padding, trained on ImageNet-1k only; tf_efficientnet_b0.ns_jft_in1k
# uses extra data and is deliberately left out.
# one random-init architecture per family; the checkpoint tag only picks the architecture
RANDOM_ARCHS="timm:resnet18.tv_in1k_random timm:resnet34.tv_in1k_random timm:vgg16_bn.tv_in1k_random timm:densenet121.tv_in1k_random timm:mobilenetv3_large_100.ra_in1k_random timm:efficientnet_b0.ra_in1k_random timm:resnet101.tv_in1k_random timm:wide_resnet50_2.tv_in1k_random timm:resnext50_32x4d.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random"
SEEDS=${SEEDS:-"1 2 3"}

for s in $SEEDS; do
  echo "=== random seed $s ===" >&2
  $PY -m layerspec.run $COMMON --seed $s --models $RANDOM_ARCHS --out $OUT/seed$s
done
echo "=== trained checkpoints ===" >&2
$PY -m layerspec.run $COMMON --seed 0 --models $TRAINED --out $OUT/trained
echo "=== done ===" >&2
