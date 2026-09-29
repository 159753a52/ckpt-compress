#!/usr/bin/env bash
export DACP_CODE=/home/gridtest/dacp_probe_code/122a989
source /home/gridtest/dacp_probe_code/122a989/experiments/scripts/remote/gpu7_env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /home/gridtest/dacp_probe_code/122a989
echo "code $(cat DACP_CODE_COMMIT) start $(date -u +%FT%TZ) threads 16"
start=$(date +%s)
/usr/bin/time -v $PYTHON experiments/scripts/run_system_bytes_comparison.py   --manifest experiments/configs/probe_pythia410m_campaign.yaml --workload pythia_410m_alpaca   --checkpoint-root $DACP_ASSETS/checkpoints --data-root $DACP_ASSETS/data --device cpu   --output /home/gridtest/dacp_probe/tolerance/system_bytes/tol_R1_fixed_s42.json --insh-bins= --seeds 42 --alphas= --ours= --extra-sparsities 0.84,0.86,0.88,0.9,0.92 --extra-alpha 1e-2 --extra-ours iterative_ucb,first_order
echo "exit $? after $(( $(date +%s) - start )) s"
