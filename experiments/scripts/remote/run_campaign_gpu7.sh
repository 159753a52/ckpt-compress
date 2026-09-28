#!/usr/bin/env bash
# Run one suite of a paper manifest on GPU 7 into a durable output root.
# Usage: run_campaign_gpu7.sh <manifest> <suite_name> [runner args]
# Output: $DACP_CAMPAIGN_ROOT/<suite_name> (default root
# /home/gridtest/dacp_probe/campaign). An existing suite there is resumed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gpu7_env.sh"
manifest="$1"
suite="$2"
shift 2
root="${DACP_CAMPAIGN_ROOT:-/home/gridtest/dacp_probe/campaign}"
out="$root/$suite"
mkdir -p "$out"
cd "$DACP_CODE"
args=(
  experiments/scripts/run_paper_experiments.py
  --manifest "$manifest"
  --device cuda:0
  --output-dir "$out"
  --checkpoint-root "$DACP_ASSETS/checkpoints"
  --data-root "$DACP_ASSETS/data"
)
if [[ -f "$out/suite_manifest.json" ]]; then
  args+=(--resume)
fi
args+=("$@")
"$PYTHON" "${args[@]}" --dry-run > "$out/plan.json"
cp "$manifest" "$out/"
log="$out/run_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "code $DACP_CODE commit $(cat DACP_CODE_COMMIT 2>/dev/null) suite $suite args: $*" | tee "$log"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader -i 7 | tee -a "$log"
start=$(date +%s)
gpu7_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 7)
# Sample every process's memory on GPU 7 every 20 s (UTC time, uuid, pid, MiB).
( while true; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader \
      | grep "$gpu7_uuid" | sed "s/^/$(date -u +%H:%M:%S) /" >> "$out/gpu7_mem.log"
    sleep 20
  done ) &
sampler=$!
set +e
"$PYTHON" "${args[@]}" 2>&1 | tee -a "$log"
status=${PIPESTATUS[0]}
set -e
kill "$sampler" 2>/dev/null || true
echo "exit $status after $(( $(date +%s) - start )) s" | tee -a "$log"
exit "$status"
