#!/bin/bash
# Start (or resume) every active Mac queue after a reboot.  Each queue skips finished runs, resumes the
# interrupted one from its resume.pt, and waits for the queues ahead of it, so this is safe to run any time.
#   bash /Users/ccli/Downloads/effective-width-claude/results/mac_queues_start.sh
cd "$(dirname "$0")/.." || exit 1
for q in p2e p2f p2g p2h p2i p2j p2k; do
  if pgrep -f "results/mac_queue_$q\.sh" > /dev/null 2>&1; then echo "$q already running"; continue; fi
  nohup caffeinate -i results/mac_queue_$q.sh >> results/mac_queue_$q.log 2>&1 &
  echo "started $q"
done
sleep 3; echo "queues alive: $(pgrep -f 'results/mac_queue_p2' | wc -l | tr -d ' ')"
