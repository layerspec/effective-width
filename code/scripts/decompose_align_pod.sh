#!/usr/bin/env bash
# P9.10 (analysis-plan 9.6): rho_align on every ImageNet checkpoint of the paper and its
# random initialisation, 6,400-image subset, 48 positions (the paper's decompose setting).
# Output: results/decompose_align/<name>_decompose.csv (checkpoint_analysis section 17 scans it).
# Usage on the pod:  cd code && PY=python bash scripts/decompose_align_pod.sh /data/imagenet_val_6400 cuda
set -u
PY=${PY:-python}
DATA=${1:-../data/imagenet_val_6400}
DEV=${2:-cuda}
OUT=../results/decompose_align; mkdir -p $OUT
MODELS="vgg16_bn resnet18 resnet50 convnext_tiny timm:vgg16.tv_in1k timm:resnet34.a1_in1k timm:resnet50.tv_in1k timm:resnet50.tv2_in1k timm:resnet50.a1_in1k timm:resnet50.a3_in1k timm:resnet50.gluon_in1k timm:resnet50.fb_ssl_yfcc100m_ft_in1k timm:resnet101.tv_in1k timm:wide_resnet50_2.tv_in1k timm:resnext50_32x4d.tv_in1k timm:densenet121.tv_in1k timm:mobilenetv2_100.ra_in1k timm:mobilenetv3_large_100.ra_in1k timm:efficientnet_b0.ra_in1k timm:convnext_tiny.fb_in22k_ft_in1k"
for m in $MODELS; do
  for v in "$m" "${m}_random"; do
    name=$(echo "$v" | tr ':' '_')
    [ -f "$OUT/${name}_decompose.csv" ] && continue
    echo "=== $v $(date) ==="
    $PY scripts/decompose_check.py --model "$v" --data $DATA --device $DEV --workers 2 --positions 48 --seed 0 --out $OUT || echo "!! $v failed"
  done
done
echo "=== decompose_align done $(date) ==="
