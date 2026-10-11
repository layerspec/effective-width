#!/bin/bash
# plan 9.56: counterfactual predictions from the stored none arms (D5 / a2), computed before any whitening arm is trained
cd /Users/ccli/Downloads/effective-width-claude/code
PY=~/.venvs/effwidth/bin/python
for s in 0 1 2; do
  $PY scripts/t3_counterfactual.py --cifar-arch vgg16_bn --checkpoint ../results/round3_pt/a5_vgg16_bn_none_s$s/vgg16_bn_cifar10_final.pt --data ../data --device cpu --positions 16 --out ../results/t3/counterfactual_vgg16_bn_none_s$s.csv
  $PY scripts/t3_counterfactual.py --cifar-arch resnet_basic52 --checkpoint ../results/round3_pt/a2_basic52_s$s/resnet_basic52_cifar10_final.pt --data ../data --device cpu --positions 16 --out ../results/t3/counterfactual_basic52_none_s$s.csv
done
echo ALL_DONE
