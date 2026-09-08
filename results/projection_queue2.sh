#!/bin/bash
# 2026-09-08: (i) leave-one-layer-out for the A1 recipe (its layer1.0.conv3 keeps one
# direction at tau<=0.99), (ii) the remaining ResNet-50 recipes (rule 10), (iii) rerun the
# first four with per-image correctness for McNemar.  Waits for the decompose rerun.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'decompose_queue.sh' > /dev/null; do sleep 120; done
C="--data ../data/imagenet_val_6400 --device mps --positions 64 --taus 0.9 0.95 0.99 0.999 --out ../results/projection"
echo "=== A1 leave-one-layer-out $(date) ==="
$PY scripts/projection_check.py --model timm:resnet50.a1_in1k $C --skip-layers layer1.0.conv3 --tag _lolo
for m in timm:resnet50.tv2_in1k timm:resnet50.a3_in1k timm:resnet50.gluon_in1k timm:resnet50.fb_ssl_yfcc100m_ft_in1k resnet50 timm:resnet50.a1_in1k resnet18 vgg16_bn; do
  echo "=== projection $m $(date) ==="
  $PY scripts/projection_check.py --model $m $C
done
echo "=== projection queue 2 done $(date) ==="
