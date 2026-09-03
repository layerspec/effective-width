#!/usr/bin/env bash
# Full measurement run on a rented GPU box (Vast.ai / RunPod / anything with
# a CUDA image).  Assumes the pytorch base image most providers offer.
#
#   bash scripts/run_on_rented_gpu.sh /path/to/imagenet/val
#
# Instances are ephemeral.  Everything that matters lands in $OUT, which you
# must copy off the machine before terminating it -- see the last section.

set -euo pipefail

DATA="${1:-synthetic}"
OUT="${OUT:-$PWD/results}"
POSITIONS="${POSITIONS:-16}"
BATCH="${BATCH:-64}"
WORKERS="${WORKERS:-8}"

echo "=== layerspec ==="
echo "data:      $DATA"
echo "out:       $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# --- deps -------------------------------------------------------------------
python -c "import torch" 2>/dev/null || pip install -q torch torchvision
pip install -q -r requirements.txt

# --- 0. correctness ---------------------------------------------------------
# Cheap, and catches a broken install before you pay for a full pass.
echo "--- correctness tests ---"
PYTHONPATH=. python tests/test_core.py

# --- 1. smoke ---------------------------------------------------------------
echo
echo "--- smoke test (synthetic, no data needed) ---"
python -m layerspec.run \
    --models resnet18_random --data synthetic \
    --limit 128 --batch-size 32 --image-size 128 \
    --out /tmp/layerspec_smoke

# --- 2. validation gate -----------------------------------------------------
# Reproduce Garg, Panda & Roy (IEEE Access 2019) Table 2 before extending the
# measurement to architectures nobody has covered.  ~1 GPU-hour.  If this does
# not reproduce, STOP -- every downstream number would be suspect.
if [ "${SKIP_GARG:-0}" != "1" ]; then
  echo
  echo "--- validation gate: reproducing Garg et al. Table 2 ---"
  python scripts/reproduce_garg.py \
      --epochs "${GARG_EPOCHS:-100}" \
      --data-root "${CIFAR_ROOT:-./data}" \
      --out "$OUT/garg"
  echo
  echo "Read the verdict above before continuing."
  echo "Set SKIP_GARG=1 to skip this gate on a re-run."
fi

# --- 3. real run ------------------------------------------------------------
# Pretrained models first: if the instance dies partway you still have the
# main result.  The random-init controls are cheap and can be redone.
echo
echo "--- pretrained models ---"
python -m layerspec.run \
    --data "$DATA" \
    --models vgg16_bn resnet18 resnet50 convnext_tiny \
    --positions "$POSITIONS" --batch-size "$BATCH" --workers "$WORKERS" \
    --out "$OUT"

echo
echo "--- random-init controls ---"
python -m layerspec.run \
    --data "$DATA" \
    --models resnet50_random convnext_tiny_random \
    --positions "$POSITIONS" --batch-size "$BATCH" --workers "$WORKERS" \
    --out "$OUT"

# --- 4. figures -------------------------------------------------------------
echo
echo "--- figures ---"
python -m layerspec.figures --results "$OUT" --out "$OUT/figures"

# --- 5. pack ----------------------------------------------------------------
# Results are small (CSVs plus compressed spectra); the whole thing fits in a
# few tens of MB, so just tar it and scp it off.
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TARBALL="/tmp/layerspec_results_${STAMP}.tar.gz"
tar czf "$TARBALL" -C "$(dirname "$OUT")" "$(basename "$OUT")"

echo
echo "=== done ==="
echo "results:  $OUT"
echo "tarball:  $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo
echo "COPY THE TARBALL OFF THIS MACHINE BEFORE TERMINATING IT."
echo "  scp -P <port> root@<host>:$TARBALL ."
echo
echo "Then check the sample-size control before reporting anything:"
echo "  any row with n_over_C_ok=False is an artefact, not a measurement."
