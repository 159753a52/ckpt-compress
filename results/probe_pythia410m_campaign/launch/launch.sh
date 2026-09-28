#!/usr/bin/env bash
# Usage: launch.sh <code_short> <queue_file_on_remote>
# Runs each line of the queue file (a command, executed with bash -c inside the
# code dir with DACP_CODE exported) sequentially under nohup; one job at a time.
set -euo pipefail
code=/home/gridtest/dacp_probe_code/$1
queue=$2
root=/home/gridtest/dacp_probe/campaign
mkdir -p "$root"
if [[ -f "$root/queue.pid" ]] && kill -0 "$(cat "$root/queue.pid")" 2>/dev/null; then
  echo "queue already running: $(cat "$root/queue.pid")"; exit 1
fi
cp "$queue" "$root/queue_$(date -u +%Y%m%dT%H%M%SZ).txt"
cat > "$root/queue_runner.sh" <<EOF
#!/usr/bin/env bash
export DACP_CODE=$code
cd $code
while IFS= read -r line; do
  [[ -z "\$line" || "\$line" == \#* ]] && continue
  echo "[\$(date -u +%FT%TZ)] START \$line"
  bash -c "\$line" < /dev/null
  echo "[\$(date -u +%FT%TZ)] END status=\$? \$line"
done < $queue
echo "[\$(date -u +%FT%TZ)] QUEUE DONE"
EOF
nohup bash "$root/queue_runner.sh" >> "$root/queue.log" 2>&1 &
echo $! > "$root/queue.pid"
echo "launched $(cat "$root/queue.pid")"
