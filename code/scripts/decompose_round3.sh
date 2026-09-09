#!/usr/bin/env bash
# Decompose every finished round-3 CIFAR run (results/round3/<run>/*_final.pt)
# on the CIFAR-10 test split, output named after the run directory, so that
# checkpoint_analysis.py section 13 can pair penalty arms with their 'none' arm.
#   bash scripts/decompose_round3.sh [device]
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/.venvs/effwidth/bin/python}
DEV=${1:-mps}
OUT=../results/round3_decompose; mkdir -p $OUT
# PT_ROOT: where the round-3 *_final.pt live (on the Mac they were rsynced to results/round3_pt).
PT_ROOT=${PT_ROOT:-../results/round3}
for d in $PT_ROOT/*/ ../results/trajectory/ ../results/trajectory_resnet50/ ../results/trajectory_seed1/ ../results/trajectory_seed2/; do
  [ -d "$d" ] || continue
  run=$(basename "$d"); pt=$(ls "$d"/*_cifar10_final.pt 2>/dev/null | head -1)
  [ -n "$pt" ] || continue
  # skip only if the CSV exists AND already carries rho_align (plan 9.6); older CSVs are redone
  if [ -f "$OUT/${run}_decompose.csv" ] && head -1 "$OUT/${run}_decompose.csv" | grep -q rho_align; then continue; fi
  arch=$(basename "$pt" | sed -E 's/_(so|srip)?_?cifar10_final.pt$//; s/_cifar10_final.pt$//')
  echo "=== $run ($arch) ==="
  $PY scripts/decompose_check.py --cifar-arch $arch --checkpoint "$pt" --name "$run" --data ../data --device $DEV --workers 2 --positions 16 --out $OUT
done
