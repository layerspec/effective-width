#!/bin/bash
# plan 9.56 (T3): the two whitening arms, 2 archs x 3 seeds each, on a RunPod GPU.  Same recipe as D5
# (trajectory_cifar defaults: 100 epochs, OneCycle, batch 128, --gpu-data).  Usage on the pod:
#   cd /workspace/effwidth/code && bash ../results/t3/run_t3_pod.sh [vgg16_bn|resnet_basic52|all] > ../results/t3/pod.log 2>&1
# Afterwards rsync results/round3/a5_*_{nd,wo}_s* back (final .pt ~60-200 MB each) and run on the Mac:
#   decompose_check.py --cifar-arch <arch> --checkpoint <baked .pt> --name a5_<arch>_<arm>_s<seed> --out ../results/round3_decompose
#   t3_postrelu.py --cifar-arch <arch> --checkpoint <wo baked .pt> --out ../results/t3/postrelu_a5_<arch>_wo_s<seed>.csv
#   t3_judge.py --results ../results
set -u
cd "$(dirname "$0")/../../code"
PY=${PY:-/workspace/effwidth/venv/bin/python}
export OMP_NUM_THREADS=8
WHICH=${1:-all}
run_arch () {
  local arch=$1 tag=$2
  for arm in nd wo; do
    for s in 0 1 2; do
      out=../results/round3/a5_${tag}_${arm}_s${s}
      if [ -f "$out/${arch}_${arm}_cifar10_final.pt" ]; then echo "skip $out"; continue; fi
      echo "=== $arch $arm seed $s  $(date)"
      $PY scripts/trajectory_whiten.py --arm $arm --arch $arch --seed $s --device cuda --gpu-data --out $out > $out.log 2>&1
      tail -3 $out.log
    done
  done
}
case $WHICH in
  vgg16_bn) run_arch vgg16_bn vgg16_bn ;;
  resnet_basic52) run_arch resnet_basic52 basic52 ;;
  all) run_arch vgg16_bn vgg16_bn; run_arch resnet_basic52 basic52 ;;
esac
echo ALL_DONE $(date)
