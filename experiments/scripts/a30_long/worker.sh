#!/bin/bash
# Queue worker of the A30 long-run study.
#   worker.sh <name> <queue.tsv> [gpu|cpu]
# Takes the first task whose dependencies exist and that no live worker holds, claims it
# with an atomic mkdir on the shared disk, refreshes the claim's heartbeat every minute,
# runs it with up to four attempts (every run resumes from its saved state) and moves on.
# A claim whose heartbeat is older than 15 minutes belongs to a dead worker and is taken over.
set -u
source "$(dirname "$0")/env.sh"
NAME=$1; QUEUE=$2; MODE=${3:-gpu}
mkdir -p "$LOGS" "$ROOT/status"
echo $$ > "/tmp/dacp_worker_${NAME}.pid"
STATUS="$ROOT/status/${NAME}.txt"

say() { echo "$(date '+%F %T') [$NAME] $*"; }
status() { echo "$(date '+%F %T') $*" > "$STATUS"; }

claim() {
  local d="$RUNS/$1/.claim"
  mkdir -p "$RUNS/$1"
  if mkdir "$d" 2>/dev/null; then echo "$NAME" > "$d/owner"; date +%s > "$d/heartbeat"; return 0; fi
  local last now; last=$(cat "$d/heartbeat" 2>/dev/null || echo 0); now=$(date +%s)
  if (( now - last > 900 )); then
    say "taking over stale claim of $1 (owner $(cat "$d/owner" 2>/dev/null))"
    rm -rf "$d"
    if mkdir "$d" 2>/dev/null; then echo "$NAME" > "$d/owner"; date +%s > "$d/heartbeat"; return 0; fi
  fi
  return 1
}

heartbeat() { while true; do date +%s > "$1/heartbeat" 2>/dev/null; sleep 60; done; }

gpu_busy_by_others() {
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  [[ -n "$used" ]] && (( used > 1500 ))
}

say "worker started (mode $MODE, queue $QUEUE)"
while true; do
  ran=0; pending=0
  while IFS=$'\t' read -r id deps cmd; do
    [[ -z "$id" || "$id" == \#* ]] && continue
    [[ -e "$RUNS/$id/DONE" ]] && continue
    pending=1
    [[ -e "$RUNS/$id/FAILED" ]] && continue
    ok=1
    for dep in ${deps//,/ }; do
      [[ "$dep" == "-" ]] && continue
      [[ -e "$ROOT/$dep" ]] || { ok=0; break; }
    done
    (( ok )) || continue
    if [[ "$MODE" == gpu ]] && gpu_busy_by_others; then
      say "GPU memory in use by another process; waiting"; status "waiting: GPU busy"; sleep 300; ran=1; break
    fi
    claim "$id" || continue
    heartbeat "$RUNS/$id/.claim" & HB=$!
    logf="$LOGS/${id//\//__}.log"
    status "running $id"
    say "start $id"
    for attempt in 1 2 3 4; do
      echo "=== $(date '+%F %T') $NAME attempt $attempt" >> "$logf"
      bash -c "$cmd" < /dev/null >> "$logf" 2>&1
      rc=$?
      [[ -e "$RUNS/$id/DONE" ]] && break
      say "$id attempt $attempt exited with $rc"
      sleep 60
    done
    kill "$HB" 2>/dev/null; wait "$HB" 2>/dev/null
    if [[ -e "$RUNS/$id/DONE" ]]; then say "done $id"; else touch "$RUNS/$id/FAILED"; say "FAILED $id"; fi
    rm -rf "$RUNS/$id/.claim"
    ran=1
    break
  done < "$QUEUE"
  if (( ! pending )); then say "queue finished"; status "finished"; break; fi
  if (( ! ran )); then status "idle: waiting for dependencies"; sleep 120; fi
done
