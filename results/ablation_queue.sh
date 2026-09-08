#!/bin/bash
# 2026-09-08: ablations A11-A15 on the Mac, after projection_queue2.
#  bn      : BN outputs alongside conv outputs (ResNet-50 V1, VGG-16 BN, random ResNet-50)
#  pos4/64 : 4 and 64 sampled positions per image instead of 16
#  resize  : direct Resize(224,224) instead of resize-256 + centre-crop-224
#  noise   : 1/f (pink) and white Gaussian inputs, trained and random ResNet-50
#  coco    : COCO-trained detection backbones on COCO val2017 images (3,200), and the
#            ImageNet ResNet-50 on the same COCO images (same weights, different data)
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'projection_queue2.sh|decompose_queue.sh' > /dev/null; do sleep 120; done
D=../data/imagenet_val_6400; O=../results/ablation
C="--device mps --batch-size 32 --workers 2 --seed 0"
echo "=== bn $(date) ===";     $PY -m layerspec.run $C --data $D --positions 16 --bn --models resnet50 vgg16_bn resnet50_random --out $O/bn
echo "=== pos4 $(date) ===";   $PY -m layerspec.run $C --data $D --positions 4  --models resnet50 --out $O/pos4
echo "=== pos64 $(date) ===";  $PY -m layerspec.run $C --data $D --positions 64 --models resnet50 --out $O/pos64
echo "=== resize $(date) ==="; $PY -m layerspec.run $C --data $D --positions 16 --preprocess resize --models resnet50 --out $O/resize
echo "=== noise $(date) ===";  $PY -m layerspec.run $C --data gaussian  --limit 6400 --positions 16 --models resnet50 resnet50_random --out $O/gaussian
                              $PY -m layerspec.run $C --data synthetic --limit 6400 --positions 16 --models resnet50 resnet50_random --out $O/pink
while [ ! -f ../data/coco/COUNT ]; do sleep 120; done
echo "=== coco $(date) ===";   $PY -m layerspec.run $C --data ../data/coco/val2017 --limit 3200 --positions 16 --bn --models tvdet:fasterrcnn_resnet50_fpn tvdet:maskrcnn_resnet50_fpn resnet50 --out $O/coco
echo "=== ablation queue done $(date) ==="
