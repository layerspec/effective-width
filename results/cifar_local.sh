#!/bin/bash
# The three CIFAR-10 trajectories on the Mac, spread over several days.
# Run this once per day (it is safe to run again at any time): each run resumes
# from its resume.pt, finished runs are skipped, and it stops when all three
# are done.  It waits for any other results/*_queue.sh to finish first.
#
#   cd /Users/ccli/Downloads/effective-width-claude && nohup caffeinate -i results/cifar_local.sh >> results/cifar_local.log 2>&1 &
#
# Closing the lid stops it; nothing is lost beyond the current epoch.
cd /Users/ccli/Downloads/effective-width-claude/code
PY=$HOME/.venvs/effwidth/bin/python
while pgrep -f 'results/[a-z_0-9]*_queue[0-9]*.sh' > /dev/null; do sleep 120; done
run() {
  local out=$1; shift
  if [ -f "$out/epoch_100_layers.csv" ]; then echo "=== $out done, skipping ==="; return; fi
  echo "=== $out start/resume $(date) ==="
  $PY scripts/trajectory_cifar.py --device mps --epochs 100 --workers 2 "$@" --out $out
}
run ../results/trajectory_resnet50 --arch resnet50 --seed 0
run ../results/trajectory_seed1    --arch vgg16_bn --seed 1
run ../results/trajectory_seed2    --arch vgg16_bn --seed 2
echo "=== all three trajectories done $(date) ==="
