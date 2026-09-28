#!/usr/bin/env bash
# Shared environment for jobs on the shared remote host (source, do not run).
# Only physical GPU 7 is used, and the process-level CUDA allocator is capped
# through sitecustomize.py (DACP_GPU_MEM_FRACTION) so this job stays far below
# the memory used by the concurrent large-model job on the same device.
export CUDA_VISIBLE_DEVICES=7
export DACP_GPU_MEM_FRACTION="${DACP_GPU_MEM_FRACTION:-0.14}"
export DACP_ASSETS="${DACP_ASSETS:-/home/gridtest/dacp_probe/assets}"
export DACP_CODE="${DACP_CODE:-/home/gridtest/dacp_probe_code/9f2d62f}"
export PYTHON="${PYTHON:-/home/gridtest/py310/bin/python}"
export PYTHONPATH="$DACP_ASSETS/pyshim"
export DACP_MODEL_ROOT="$DACP_ASSETS/models"
export DACP_DATA_ROOT="$DACP_ASSETS/data"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME="$DACP_ASSETS/hf_home"
export PYTHONUNBUFFERED=1
