#!/bin/bash
# Laptop-side watchdog of the A30 long runs.
#   watch_local.sh <minutes> [interval_s]
# Every interval: make sure the workers on a30-1/2/3 are running, regenerate the report on
# a30-1 and copy results, logs and reports (no model states) into the local results folder.
# Exits after <minutes> or when every task is done (exit 0); a FAILED task is only logged.
MINUTES=${1:-90}; INTERVAL=${2:-600}
ROOT=/lihongliang/fangzl/dacp-a30
DEST=/d/Paper/EMNLP_26_reorganized/code/checkpoint_compress/results/a30_long_20261121
mkdir -p "$DEST"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20"
quiet() { grep -v -E "WARNING|vulnerable|upgraded"; }
end=$(( $(date +%s) + MINUTES * 60 ))
while :; do
  now=$(date '+%F %T')
  for h in a30-1 a30-2 a30-3; do
    timeout 60 $SSH $h "bash $ROOT/code/ensure_worker.sh gpu-$h $ROOT/code/queue_chain.tsv gpu" 2>&1 | quiet | sed "s/^/[$now $h] /"
  done
  timeout 60 $SSH a30-2 "[ -e $ROOT/runs/prep_data/DONE ] || bash $ROOT/code/ensure_worker.sh cpu-a30-2 $ROOT/code/queue_cpu.tsv cpu" 2>&1 | quiet | sed "s/^/[$now a30-2] /"
  timeout 300 $SSH a30-1 "source $ROOT/code/env.sh && \$PY $ROOT/code/report.py --runs $ROOT/runs --out $ROOT/reports > /dev/null 2> $ROOT/reports/report.err; cd $ROOT && tar czf - --exclude='*.pt' --exclude='*.tmp*' --exclude='.claim' runs logs reports status code/queue.tsv data/fineweb_edu/info.json 2>/dev/null" 2>/dev/null | tar xzf - -C "$DEST" 2>/dev/null
  st=$(timeout 60 $SSH a30-1 "cd $ROOT/runs; echo done=\$(ls -d AC_*/DONE BC_*/DONE BO_*/DONE 2>/dev/null | wc -l) failed=\$(ls -d */FAILED B_pilot/*/FAILED 2>/dev/null | tr '\n' ' '); cat $ROOT/status/*.txt | tr '\n' ';'; total=\$(grep -vc '^#' $ROOT/code/queue_chain.tsv); echo total=\$total" 2>&1 | quiet | tr '\n' ' ')
  echo "[$now] $st"
  # a failed task is reported but the watch goes on (the other tasks keep running and need syncing)
  if [[ "$st" =~ failed=[A-Za-z] ]]; then echo "[$now] a task FAILED"; fi
  d=$(echo "$st" | sed -n 's/.*done=\([0-9]*\).*/\1/p'); t=$(echo "$st" | sed -n 's/.*total=\([0-9]*\).*/\1/p')
  if [[ -n "$d" && -n "$t" && "$d" -ge "$t" ]]; then echo "[$now] all $t tasks done"; exit 0; fi
  (( $(date +%s) >= end )) && { echo "[$now] watch window over"; exit 0; }
  sleep "$INTERVAL"
done
