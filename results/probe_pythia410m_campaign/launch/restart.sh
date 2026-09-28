#!/usr/bin/env bash
# Stop my queue runner (and its children) and remove the partial main_k1 suite so it restarts cleanly.
root=/home/gridtest/dacp_probe/campaign
qp=$(cat "$root/queue.pid")
echo "queue pid $qp"
pkill -TERM -P "$qp" 2>/dev/null
for c in $(pgrep -f "run_paper_experiments.py --manifest experiments/configs/probe_pythia410m_campaign.yaml"); do
  echo "kill runner $c"; kill "$c"
done
kill "$qp" 2>/dev/null
sleep 5
ps -u gridtest -o pid,cmd | grep -E "run_paper_experiments|queue_runner|run_campaign" | grep -v grep
ls "$root/main_k1"
rm -rf "$root/main_k1"
rm -f "$root/queue.pid"
