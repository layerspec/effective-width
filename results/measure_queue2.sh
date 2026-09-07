#!/bin/bash
# Queued 2026-09-07 after measure_queue.sh: the other bottleneck architectures
# (ResNet-101, Wide-ResNet-50-2, ResNeXt-50) and MobileNetV2, trained recipes
# plus three random seeds each, on the 6,400-image subset.  Referee item:
# separate depth and width from block type before claiming "bottleneck".
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'trajectory_queue.sh|measure_queue.sh' > /dev/null; do sleep 120; done
COMMON="--data ../data/imagenet_val_6400 --device mps --batch-size 32 --workers 2 --positions 16"
R="timm:resnet101.tv_in1k_random timm:wide_resnet50_2.tv_in1k_random timm:resnext50_32x4d.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random"
for s in 1 2 3; do
  echo "=== bottleneck extras random seed $s $(date) ==="
  $PY -m layerspec.run $COMMON --seed $s --models $R --out ../results/local6400/archs/seed$s
done
echo "=== bottleneck extras trained $(date) ==="
$PY -m layerspec.run $COMMON --seed 0 --out ../results/local6400/archs/trained --models \
  timm:resnet101.tv_in1k timm:resnet101.tv2_in1k timm:resnet101.a1_in1k timm:resnet101.a3_in1k timm:resnet101.gluon_in1k \
  timm:wide_resnet50_2.tv_in1k timm:wide_resnet50_2.tv2_in1k timm:wide_resnet50_2.racm_in1k \
  timm:resnext50_32x4d.tv_in1k timm:resnext50_32x4d.tv2_in1k timm:resnext50_32x4d.a1_in1k timm:resnext50_32x4d.a3_in1k timm:resnext50_32x4d.gluon_in1k \
  timm:mobilenetv2_100.ra_in1k
echo "=== measure queue 2 done $(date) ==="
