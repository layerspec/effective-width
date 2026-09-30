#!/bin/bash
# Start (or resume) every active Mac queue after a reboot.  Each queue skips finished runs, resumes the
# interrupted one from its resume.pt, and waits for the queues ahead of it, so this is safe to run any time.
#   bash /Users/ccli/Downloads/effective-width-claude/results/mac_queues_start.sh
cd "$(dirname "$0")/.." || exit 1
for q in p2e p2f p2g p2h p2i p2j p2k p2l; do
  if pgrep -f "results/mac_queue_$q\.sh" > /dev/null 2>&1; then echo "$q already running"; continue; fi
  nohup caffeinate -i results/mac_queue_$q.sh >> results/mac_queue_$q.log 2>&1 &
  echo "started $q"
done
if pgrep -f "results/mac_measure_bplus\.sh" > /dev/null 2>&1; then echo "measure_bplus already running"; else
  nohup caffeinate -i results/mac_measure_bplus.sh >> results/mac_measure_bplus.log 2>&1 &
  echo "started measure_bplus"
fi
if pgrep -f "results/mac_measure_bplus2\.sh" > /dev/null 2>&1; then echo "measure_bplus2 already running"; else
  nohup caffeinate -i results/mac_measure_bplus2.sh >> results/mac_measure_bplus2.log 2>&1 &
  echo "started measure_bplus2"
fi
if pgrep -f "results/mac_measure_bplus3\.sh" > /dev/null 2>&1; then echo "measure_bplus3 already running"; else
  nohup caffeinate -i results/mac_measure_bplus3.sh >> results/mac_measure_bplus3.log 2>&1 &
  echo "started measure_bplus3"
fi
if pgrep -f "results/mac_measure_bplus4\.sh" > /dev/null 2>&1; then echo "measure_bplus4 already running"; else
  nohup caffeinate -i results/mac_measure_bplus4.sh >> results/mac_measure_bplus4.log 2>&1 &
  echo "started measure_bplus4"
fi
if pgrep -f "results/mac_measure_bplus5\.sh" > /dev/null 2>&1; then echo "measure_bplus5 already running"; else
  nohup caffeinate -i results/mac_measure_bplus5.sh >> results/mac_measure_bplus5.log 2>&1 &
  echo "started measure_bplus5"
fi
sleep 3; echo "queues alive: $(pgrep -f 'results/mac_queue_p2' | wc -l | tr -d ' ')"
