#!/bin/bash
# 2026-10-13 (plan 9.56, T3 four arms): the RunPod balance ran out (US$0.93 left at 23 h) with VGG wo s2 at epoch 70 and the
# six resnet_basic52 arms not started.  This queue finishes them on the Mac, free: resume VGG wo s2 from the resume.pt pulled
# off the pod, then basic52 nd/wo x seeds 0-2; after each run decompose_check + (wo) t3_postrelu, then t3_judge.
# Same recipe as the Mac-side D5 arms (mac_queue_p2j: --device mps --workers 2, no --gpu-data).  Resumable: finished runs skipped.
# Waits for plan 9.57 (m2_activation) to finish first (one MPS job at a time), and for the pod to be gone
# (results/t3/pod_gone, written by the puller) so the same resume.pt is never continued in two places.
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
export OMP_NUM_THREADS=4
echo "=== mac_queue_t3 start $(date) ==="
until [ -f ../results/m2_activation/judgement.txt ]; do
  until pgrep -f "m2_activation_se[e]ds" > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 300; done
  [ -f ../results/m2_activation/judgement.txt ] || { echo "m2_activation not running and no judgement.txt $(date)"; break; }
done
until [ -f ../results/t3/pod_gone ] || [ -f ../results/round3/a5_vgg16_bn_wo_s2/vgg16_bn_wo_cifar10_final.pt ]; do sleep 300; done
echo "=== gates open $(date) ==="
run_one () {
  local arch=$1 tag=$2 arm=$3 s=$4
  local out=../results/round3/a5_${tag}_${arm}_s${s}
  local ckpt=$out/${arch}_${arm}_cifar10_final.pt
  if [ ! -f "$ckpt" ]; then
    echo "=== $arch $arm seed $s  $(date)"
    $PY -u scripts/trajectory_whiten.py --arm $arm --arch $arch --seed $s --device mps --workers 2 --out $out >> $out.log 2>&1
    tail -2 $out.log
  fi
  [ -f "$ckpt" ] || { echo "no final checkpoint for $out"; return; }
  [ -f ../results/round3_decompose/a5_${tag}_${arm}_s${s}_decompose.csv ] || \
    $PY scripts/decompose_check.py --cifar-arch $arch --checkpoint $ckpt --data ../data --device mps --name a5_${tag}_${arm}_s${s} --out ../results/round3_decompose
  if [ "$arm" = wo ]; then
    [ -f ../results/t3/postrelu_a5_${tag}_${arm}_s${s}.csv ] || \
      $PY scripts/t3_postrelu.py --cifar-arch $arch --checkpoint $ckpt --data ../data --device mps --out ../results/t3/postrelu_a5_${tag}_${arm}_s${s}.csv
  fi
}
for s in 0 1 2; do run_one vgg16_bn vgg16_bn nd $s; run_one vgg16_bn vgg16_bn wo $s; done
for arm in nd wo; do for s in 0 1 2; do run_one resnet_basic52 basic52 $arm $s; done; done
$PY scripts/t3_judge.py --results ../results
echo "=== mac_queue_t3 done $(date) ==="
