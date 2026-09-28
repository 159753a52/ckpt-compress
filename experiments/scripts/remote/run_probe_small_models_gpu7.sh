#!/usr/bin/env bash
# Run experiments/scripts/run_probe_small_models.sh on GPU 7 with remapped
# checkpoint/data roots. Usage: run_probe_small_models_gpu7.sh <workload> [runner args]
# The code tree's results/paper_runs/probe_small_models is expected to be a
# symlink to the durable output root.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/gpu7_env.sh"
workload="$1"
shift
cd "$DACP_CODE"
CHECKPOINT_ROOT="$DACP_ASSETS/checkpoints" DATA_ROOT="$DACP_ASSETS/data" \
  bash experiments/scripts/run_probe_small_models.sh 7 "$workload" "$@"
