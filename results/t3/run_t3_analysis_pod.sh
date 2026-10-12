#!/bin/bash
# T3 analysis of the finished arms on the pod (Mac is busy with plan 9.57); same scripts as the Mac path, --device cuda.
cd /workspace/effwidth/code || exit 1
PY=/workspace/effwidth/venv/bin/python
export OMP_NUM_THREADS=8
mkdir -p ../results/round3_decompose
for tag in vgg16_bn basic52; do
  arch=$tag; [ "$tag" = basic52 ] && arch=resnet_basic52
  for arm in nd wo; do for s in 0 1 2; do
    r=${tag}_${arm}_s${s}
    ckpt=../results/round3/a5_$r/${arch}_${arm}_cifar10_final.pt
    [ -f "$ckpt" ] || continue
    if [ ! -f ../results/round3_decompose/a5_${r}_decompose.csv ]; then
      echo "=== decompose a5_$r $(date)"
      $PY scripts/decompose_check.py --cifar-arch $arch --checkpoint $ckpt --data ../data --device cuda --name a5_$r --out ../results/round3_decompose
    fi
    if [ "$arm" = wo ] && [ ! -f ../results/t3/postrelu_a5_$r.csv ]; then
      echo "=== postrelu a5_$r $(date)"
      $PY scripts/t3_postrelu.py --cifar-arch $arch --checkpoint $ckpt --data ../data --device cuda --out ../results/t3/postrelu_a5_$r.csv
    fi
  done; done
done
echo ANALYSIS_DONE $(date)
