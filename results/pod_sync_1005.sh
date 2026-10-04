#!/bin/bash
# 2026-10-02: pull the pod's A31 / A29b results (no weights) every 15 minutes until the pod is gone (it terminates itself
# after the A31 end cells, /workspace/effwidth/terminate_after_a31.sh) or the Mac is shut down.  No Claude involved.
R=/Users/ccli/Downloads/effective-width-claude/results
SSH="ssh -p 10224 -i $HOME/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=20"
fail=0
while [ $fail -lt 8 ]; do
  ok=1
  for d in a31 a29b; do
    rsync -az --exclude '*.pt' --exclude '*.pth' -e "$SSH" root@203.57.40.67:/workspace/effwidth/results/$d/ $R/$d/ < /dev/null || ok=0
  done
  rsync -az -e "$SSH" root@203.57.40.67:/workspace/effwidth/results/terminate.log root@203.57.40.67:/workspace/effwidth/results/chain_1005.log $R/ < /dev/null || ok=0
  if [ $ok -eq 1 ]; then fail=0; echo "synced $(date) : $(ls $R/a31/*/result.json 2>/dev/null | wc -l | tr -d ' ') A31 result.json"; else fail=$((fail + 1)); echo "sync failed ($fail) $(date)"; fi
  sleep 900
done
echo "pod unreachable 8 times in a row; stopping $(date)"
