#!/usr/bin/env bash
# Usage: sys_cpu.sh <code_short> <out_name> [driver args]  -- CPU-only (no GPU visible), nohup
set -euo pipefail
code=/home/gridtest/dacp_probe_code/$1
name=$2
shift 2
out=/home/gridtest/dacp_probe/campaign/system_bytes
mkdir -p "$out"
cat > "$out/$name.sh" <<EOF
#!/usr/bin/env bash
export DACP_CODE=$code
source $code/experiments/scripts/remote/gpu7_env.sh
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=24 MKL_NUM_THREADS=24 TOKENIZERS_PARALLELISM=false
cd $code
echo "code \$(cat DACP_CODE_COMMIT) start \$(date -u +%FT%TZ)"
start=\$(date +%s)
/usr/bin/time -v \$PYTHON experiments/scripts/run_system_bytes_comparison.py \
  --manifest experiments/configs/probe_pythia410m_campaign.yaml --workload pythia_410m_alpaca \
  --checkpoint-root \$DACP_ASSETS/checkpoints --data-root \$DACP_ASSETS/data --device cpu \
  --output $out/$name.json $*
echo "exit \$? after \$(( \$(date +%s) - start )) s"
EOF
nohup bash "$out/$name.sh" > "$out/$name.log" 2>&1 < /dev/null &
echo "launched $! -> $out/$name.log"
