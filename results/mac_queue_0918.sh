#!/bin/bash
# 2026-09-18 evening chain (after the 13:12 shutdown): finish the COCO out-of-sample measurements first
# (nothing is training, so the pause is a no-op), then the P2 controller runs, then P2b.  Each stage is resumable.
#   cd <repo> && nohup caffeinate -i results/mac_queue_0918.sh >> results/mac_queue_0918.log 2>&1 &
set -u
cd "$(dirname "$0")/.." || exit 1
echo "=== chain 0918 start $(date) ==="
bash results/apriori_queue2.sh >> results/apriori_queue2.log 2>&1
echo "=== coco done $(date) ==="
bash results/mac_queue_p2.sh   >> results/mac_queue_p2.log 2>&1
echo "=== p2 done $(date) ==="
bash results/mac_queue_p2b.sh  >> results/mac_queue_p2b.log 2>&1
echo "=== chain 0918 all done $(date) ==="
