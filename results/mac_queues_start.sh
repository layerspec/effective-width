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
if pgrep -f "results/mac_measure_bplus6\.sh" > /dev/null 2>&1; then echo "measure_bplus6 already running"; else
  nohup caffeinate -i results/mac_measure_bplus6.sh >> results/mac_measure_bplus6.log 2>&1 &
  echo "started measure_bplus6"
fi
if pgrep -f "results/mac_queue_1001\.sh" > /dev/null 2>&1; then echo "queue_1001 already running"; else
  nohup caffeinate -i results/mac_queue_1001.sh >> results/mac_queue_1001.log 2>&1 &
  echo "started queue_1001"
fi
if pgrep -f "results/mac_queue_1005\.sh" > /dev/null 2>&1; then echo "queue_1005 already running"; else
  nohup caffeinate -i results/mac_queue_1005.sh >> results/mac_queue_1005.log 2>&1 &
  echo "started queue_1005"
fi
sleep 3; echo "queues alive: $(pgrep -f 'results/mac_queue_p2' | wc -l | tr -d ' ')"
# 2026-10-12: plan 9.57 (Theorem M.2 on the activation step); resumes from results/m2_activation/rows.csv, skips finished layers.
if [ -f results/m2_activation/judgement.txt ]; then echo "m2_activation done"; elif pgrep -f "m2_activation_se[e]ds" > /dev/null 2>&1; then echo "m2_activation already running"; else
  (cd code && nohup caffeinate -i ~/.venvs/effwidth/bin/python scripts/m2_activation_seeds.py --device mps --limit 1600 --positions 32 --draws 40 --resume >> ../results/m2_activation.log 2>&1 < /dev/null &)
  echo "started m2_activation (resume)"
fi
# 2026-10-13: plan 9.56 T3 arms moved to the Mac after the pod balance ran out; waits for m2_activation and for results/t3/pod_gone.
if [ -f results/t3/judgement.txt ]; then echo "t3 queue done"; elif pgrep -f "results/mac_queue_t3\.sh" > /dev/null 2>&1; then echo "t3 queue already running"; else
  nohup caffeinate -i results/mac_queue_t3.sh >> results/mac_queue_t3.log 2>&1 < /dev/null &
  echo "started t3 queue"
fi
