#!/usr/bin/env bash
# The 2x2 that separates class count from head depth as explanations for the
# terminal collapse Garg et al. report and we did not reproduce on ImageNet.
#
#   {CIFAR-10, CIFAR-100} x {small head, deep head}
#
# The convolutional trunk is identical in all four cells, so a difference in
# its profile is attributable to the factor that was varied.  Read the result
# off the four "2x2 cell" blocks at the end of the log:
#
#   k*/C at features.40 tracks the DATASET  -> class count (neural collapse)
#   ... tracks the HEAD                     -> head depth
#   ... tracks both                         -> say so; do not pick one
#
# The (cifar10, small) cell is additionally the validation gate against their
# published Table 2; the other three are the contrast and are not comparable
# with it.
#
# Cost: about one GPU-hour per cell on a 3090, four total, plus seconds each
# for the measurement.  CIFAR downloads itself.
#
# Usage:  bash code/scripts/run_cifar_2x2.sh [OUT_DIR] [EPOCHS]

set -euo pipefail

OUT="${1:-results_cifar2x2}"
EPOCHS="${2:-100}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$OUT"
echo "output -> $OUT   epochs -> $EPOCHS"
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available())"

for ds in cifar10 cifar100; do
  for head in small deep; do
    tag="${ds}_${head}"
    if [ -f "$OUT/vgg16_${tag}_layers.csv" ]; then
      echo "=== $tag already done, skipping ==="
      continue
    fi
    echo
    echo "=================================================================="
    echo "=== $tag ==="
    echo "=================================================================="
    # Not `set -e`-fatal: one cell failing should not discard the other three.
    if ! python3 "$HERE/scripts/reproduce_garg.py" \
        --dataset "$ds" --head "$head" \
        --epochs "$EPOCHS" --out "$OUT" 2>&1 | tee "$OUT/${tag}.log"; then
      echo "!! $tag FAILED -- see $OUT/${tag}.log; continuing with the rest"
    fi
  done
done

echo
echo "=================================================================="
echo "=== the 2x2, last conv layer (features.40, C=512) ==="
echo "=================================================================="
grep -h -A9 "2x2 cell" "$OUT"/*.log || echo "(no cells completed)"
echo
echo "Copy $OUT back before terminating the machine:"
echo "  tar czf cifar2x2.tar.gz $OUT"
