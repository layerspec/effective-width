#!/bin/bash
# P2k (2026-09-23, author: every quality item Mac time can buy; hostile review 09-20 "transformer arm"):
# the width controller on a transformer.  CIFAR-10 ViT (patch 4, dim 192, depth 6, heads 3, MLP hidden 768)
# trained from scratch with scripts/trajectory_vit.py; the six MLP hidden widths are the sites.
#   full       full width, seeds 0-2 (seed 0 first: the ruler reads it)
#   ruler      widths = k*(0.999) of the post-GELU covariance of the seed-0 full net (scripts/vit_ruler.py)
#   uniform    equal widths at the ruler's parameter count
#   b<frac>    the controller, budget = the ruler's parameter fraction, triggers 15/40
#   apriori    the controller acting at epoch 0 as well (a-priori start)
# ~3 h per run on MPS, 15 runs.  Waits for mac_queue_p2j.sh.  Resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_p2k.sh >> results/mac_queue_p2k.log 2>&1 &
set -u
cd "$(dirname "$0")/../code" || exit 1
PY=$HOME/.venvs/effwidth/bin/python
until pgrep -f 'results/mac_queue_(0918|p2c|p2d|p2e|p2f|p2g|p2h|p2i|p2j)\.sh' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done   # a failing pgrep (shutdown) counts as busy
until pgrep -f 'scripts/trajectory_(cifar|vit)\.py' > /dev/null 2>&1; [ $? -eq 1 ]; do sleep 120; done
echo "=== mac_queue_p2k start/resume $(date) ==="
OUT=../results/vit_c10
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_vit.py --device mps --epochs 100 --workers 2 --dataset cifar10 "$@" --out $out
}
run $OUT/full_s0 --seed 0
if [ ! -f $OUT/widths.json ]; then
  $PY scripts/vit_ruler.py --checkpoint $OUT/full_s0/vit_c10_cifar10_final.pt --out $OUT/widths.json --device mps
fi
RULER=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$OUT/widths.json'))['ruler']))")
UNIFORM=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('$OUT/widths.json'))['uniform']))")
FRAC=$($PY -c "import json; print(round(json.load(open('$OUT/widths.json'))['fraction']['ruler'], 3))")
echo "ruler widths: $RULER   uniform: $UNIFORM   budget fraction: $FRAC"
for s in 0 1 2; do
  run $OUT/full_s$s     --seed $s
  run $OUT/ruler_s$s    --seed $s --widths $RULER
  run $OUT/uniform_s$s  --seed $s --widths $UNIFORM
  run $OUT/b_s$s        --seed $s --realloc-budget $FRAC
  run $OUT/apriori_s$s  --seed $s --realloc-budget $FRAC --realloc-epochs 0 15 40
done
echo "=== mac_queue_p2k all done $(date) ==="
