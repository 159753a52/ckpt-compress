#!/usr/bin/env bash
export DACP_CODE=/home/gridtest/dacp_probe_code/122a989
source /home/gridtest/dacp_probe_code/122a989/experiments/scripts/remote/gpu7_env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /home/gridtest/dacp_probe_code/122a989
echo "code $(cat DACP_CODE_COMMIT) start $(date -u +%FT%TZ) threads 16"
start=$(date +%s)
/usr/bin/time -v $PYTHON experiments/scripts/run_system_bytes_comparison.py   --manifest experiments/configs/probe_pythia410m_campaign.yaml --workload pythia_410m_alpaca   --checkpoint-root $DACP_ASSETS/checkpoints --data-root $DACP_ASSETS/data --device cpu   --output /home/gridtest/dacp_probe/tolerance/system_bytes/tol_R1_alphas_s42.json --insh-bins= --seeds 42 --alphas 6e-4,7e-4,8e-4,9e-4,1e-3 --ours dacp,iterative_ucb
echo "exit $? after $(( $(date +%s) - start )) s"
