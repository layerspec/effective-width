#!/bin/bash
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
for m in timm:vgg16_bn.tv_in1k_random timm:resnet18.tv_in1k_random; do
  for inp in matched real; do
    echo "=== $m $inp $(date) ==="
    $PY scripts/apriori_relu_check.py $m --inputs $inp --seed 1 2>&1 | grep -v "Warning\|super()"
  done
done
echo "=== relu queue done $(date) ==="
