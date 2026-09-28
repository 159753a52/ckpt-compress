#!/usr/bin/env bash
export DACP_CODE=/home/gridtest/dacp_probe_code/abfbd4d
source /home/gridtest/dacp_probe_code/abfbd4d/experiments/scripts/remote/gpu7_env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false
cd /home/gridtest/dacp_probe_code/abfbd4d
echo "code $(cat DACP_CODE_COMMIT) start $(date -u +%FT%TZ) threads 16"
start=$(date +%s)
/usr/bin/time -v $PYTHON experiments/scripts/run_system_bytes_comparison.py   --manifest experiments/configs/probe_pythia410m_campaign.yaml --workload pythia_410m_alpaca   --checkpoint-root $DACP_ASSETS/checkpoints --data-root $DACP_ASSETS/data --device cpu   --output /home/gridtest/dacp_probe/campaign/system_bytes_iter/iter_R1_alphas_s43.json --insh-bins= --seeds 43 --alphas 1e-5,5e-5,2e-4,4e-4,6e-4,1e-3,1e-2 --ours iterative_first_order,iterative_ucb --skip-excp-record
echo "exit $? after $(( $(date +%s) - start )) s"
