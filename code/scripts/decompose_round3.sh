#!/usr/bin/env bash
# Decompose every finished round-3 CIFAR run (results/round3/<run>/*_final.pt)
# on the CIFAR-10 test split, output named after the run directory, so that
# checkpoint_analysis.py section 13 can pair penalty arms with their 'none' arm.
#   bash scripts/decompose_round3.sh [device]
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/effwidth/bin/python}
DEV=${1:-mps}
OUT=../results/round3_decompose; mkdir -p $OUT
for d in ../results/round3/*/ ../results/trajectory/ ../results/trajectory_resnet50/ ../results/trajectory_seed1/ ../results/trajectory_seed2/; do
  [ -d "$d" ] || continue
  run=$(basename "$d"); pt=$(ls "$d"/*_cifar10_final.pt 2>/dev/null | head -1)
  [ -n "$pt" ] || continue
  [ -f "$OUT/${run}_decompose.csv" ] && continue
  arch=$(basename "$pt" | sed -E 's/_(so|srip)?_?cifar10_final.pt$//; s/_cifar10_final.pt$//')
  echo "=== $run ($arch) ==="
  $PY scripts/decompose_check.py --cifar-arch $arch --checkpoint "$pt" --name "$run" --data ../data --device $DEV --workers 2 --positions 16 --out $OUT
done
