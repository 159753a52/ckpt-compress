#!/bin/bash
# Start the worker <name> on this container unless it is already running.
#   ensure_worker.sh <name> <queue.tsv> [gpu|cpu]
NAME=$1; QUEUE=$2; MODE=${3:-gpu}
DIR="$(cd "$(dirname "$0")" && pwd)"
PIDF="/tmp/dacp_worker_${NAME}.pid"
if [[ -f "$PIDF" ]]; then
  pid=$(cat "$PIDF")
  if kill -0 "$pid" 2>/dev/null && grep -q "worker.sh" "/proc/$pid/cmdline" 2>/dev/null; then
    echo "running $NAME pid $pid"
    exit 0
  fi
fi
source "$DIR/env.sh"
mkdir -p "$LOGS"
setsid nohup bash "$DIR/worker.sh" "$NAME" "$QUEUE" "$MODE" >> "$LOGS/worker_${NAME}.log" 2>&1 < /dev/null &
sleep 1
echo "started $NAME pid $(cat "$PIDF" 2>/dev/null)"
