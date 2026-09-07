#!/bin/bash
# Rerun of measure_queue2.sh that skips checkpoints already measured (written
# 2026-09-07 15:55 in case the first run is interrupted at the end of the day).
#   nohup caffeinate -i results/measure_queue2_resume.sh >> results/measure_queue2.log 2>&1 &
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
COMMON="--data ../data/imagenet_val_6400 --device mps --batch-size 32 --workers 2 --positions 16"
T=../results/local6400/archs/trained
for m in timm:resnet101.tv_in1k timm:resnet101.tv2_in1k timm:resnet101.a1_in1k timm:resnet101.a3_in1k timm:resnet101.gluon_in1k \
         timm:wide_resnet50_2.tv_in1k timm:wide_resnet50_2.tv2_in1k timm:wide_resnet50_2.racm_in1k \
         timm:resnext50_32x4d.tv_in1k timm:resnext50_32x4d.tv2_in1k timm:resnext50_32x4d.a1_in1k timm:resnext50_32x4d.a3_in1k timm:resnext50_32x4d.gluon_in1k \
         timm:mobilenetv2_100.ra_in1k; do
  f=$T/$(echo $m | sed 's/:/_/')_layers.csv
  if [ -f "$f" ]; then echo "=== $m already measured ==="; continue; fi
  echo "=== $m (resume) $(date) ==="
  $PY -m layerspec.run $COMMON --seed 0 --models $m --out $T
done
for s in 1 2 3; do
  for m in timm:resnet101.tv_in1k_random timm:wide_resnet50_2.tv_in1k_random timm:resnext50_32x4d.tv_in1k_random timm:mobilenetv2_100.ra_in1k_random; do
    f=../results/local6400/archs/seed$s/$(echo $m | sed 's/:/_/')_layers.csv
    [ -f "$f" ] || { echo "=== $m seed $s (resume) ==="; $PY -m layerspec.run $COMMON --seed $s --models $m --out ../results/local6400/archs/seed$s; }
  done
done
echo "=== measure queue 2 done $(date) ==="
