#!/usr/bin/env bash
export DACP_CODE=/home/gridtest/dacp_probe_code/f864e1b
source /home/gridtest/dacp_probe_code/f864e1b/experiments/scripts/remote/gpu7_env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 TOKENIZERS_PARALLELISM=false
cd /home/gridtest/dacp_probe_code/f864e1b
echo "code $(cat DACP_CODE_COMMIT) start $(date -u +%FT%TZ)"
start=$(date +%s)
/usr/bin/time -v $PYTHON experiments/scripts/run_system_bytes_comparison.py   --manifest experiments/configs/probe_pythia410m_campaign.yaml --workload pythia_410m_alpaca   --checkpoint-root $DACP_ASSETS/checkpoints --data-root $DACP_ASSETS/data --device cpu   --output /home/gridtest/dacp_probe/campaign/system_bytes/smoke.json --seeds 42 --alphas 1e-3 --ours probe_topk_m1_exact_global --insh-bins 16
echo "exit $? after $(( $(date +%s) - start )) s"
