#!/usr/bin/env bash
export DACP_CODE=/home/gridtest/dacp_probe_code/0d39433
source /home/gridtest/dacp_probe_code/0d39433/experiments/scripts/remote/gpu7_env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false
cd /home/gridtest/dacp_probe_code/0d39433
echo "code $(cat DACP_CODE_COMMIT) start $(date -u +%FT%TZ)"
start=$(date +%s)
/usr/bin/time -v $PYTHON experiments/scripts/run_system_bytes_comparison.py   --manifest experiments/configs/probe_pythia410m_campaign.yaml --workload pythia_410m_alpaca   --checkpoint-root $DACP_ASSETS/checkpoints --data-root $DACP_ASSETS/data --device cpu   --output /home/gridtest/dacp_probe/campaign/system_bytes/insh_p0_s42.json --seeds 42 --alphas= --ours= --insh-bins 16,64,254 --insh-prune 0.0
echo "exit $? after $(( $(date +%s) - start )) s"
