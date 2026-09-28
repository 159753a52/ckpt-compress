#!/usr/bin/env bash
# Run experiments/configs/probe_small_models.yaml for one workload on one GPU.
#
# Usage: bash experiments/scripts/run_probe_small_models.sh <gpu> <workload> [extra runner args]
#   <gpu>       physical GPU index; only 5, 6 or 7 are accepted
#   <workload>  gpt2_medium_wikitext103 | pythia_410m_alpaca | pythia_1b_alpaca
# Extra arguments (e.g. --prune-ratio 0.5 --seeds 42) are passed to the runner.
# Environment: PYTHON (default python), CHECKPOINT_ROOT and DATA_ROOT remap the
# repository-relative checkpoints/ and data/ trees.
# An existing suite in the output directory is resumed with --resume.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <gpu:5|6|7> <workload> [extra runner args]" >&2
  exit 2
fi
gpu="$1"
workload="$2"
shift 2

case "$gpu" in
  5|6|7) ;;
  *) echo "refusing GPU '$gpu': only physical GPUs 5, 6, 7 are allowed" >&2; exit 2 ;;
esac

case "$workload" in
  gpt2_medium_wikitext103)
    checkpoint="checkpoints/gpt2_medium_wikitext103_1000steps/checkpoint_step_1000.pt"
    data="data/wikitext103" ;;
  pythia_410m_alpaca)
    checkpoint="checkpoints/pythia_410m_alpaca/checkpoint_step_1000.pt"
    data="data" ;;
  pythia_1b_alpaca)
    checkpoint="checkpoints/pythia_1b_alpaca/checkpoint_step_1000.pt"
    data="data" ;;
  *) echo "unknown workload '$workload'" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python_bin="${PYTHON:-python}"
manifest="experiments/configs/probe_small_models.yaml"
output_dir="results/paper_runs/probe_small_models/${workload}"

checkpoint_path="${CHECKPOINT_ROOT:-$repo_root/checkpoints}/${checkpoint#checkpoints/}"
data_path="${DATA_ROOT:-$repo_root/data}/${data#data}"
if [[ ! -f "$checkpoint_path" ]]; then
  echo "missing checkpoint: $checkpoint_path" >&2
  exit 1
fi
if [[ ! -d "$data_path" ]]; then
  echo "missing data directory: $data_path" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONUNBUFFERED=1

args=(
  experiments/scripts/run_paper_experiments.py
  --manifest "$manifest"
  --workload "$workload"
  --device cuda:0
  --output-dir "$output_dir"
)
if [[ -n "${CHECKPOINT_ROOT:-}" ]]; then
  args+=(--checkpoint-root "$CHECKPOINT_ROOT")
fi
if [[ -n "${DATA_ROOT:-}" ]]; then
  args+=(--data-root "$DATA_ROOT")
fi
if [[ -f "$output_dir/suite_manifest.json" ]]; then
  args+=(--resume)
fi
args+=("$@")

mkdir -p "$output_dir"
"$python_bin" "${args[@]}" --dry-run > "$output_dir/plan.json"
log="$output_dir/run_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "GPU $gpu (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES) workload $workload -> $output_dir" | tee "$log"
"$python_bin" "${args[@]}" 2>&1 | tee -a "$log"
