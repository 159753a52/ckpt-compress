#!/bin/bash
# Append to the chain queue of the A30 long runs (run on any container: bash add_seeds_44_45.sh):
#   - study B at the official ExCP threshold (alpha 5e-5) for seeds 44 and 45, to 7500 steps: the ExCP run,
#     the pilot-chosen DACP-G_rho deleting as many entries as that ExCP run at every checkpoint, and the
#     no-compression baseline.  Seed 42 ran to 7500 steps; seed 43 stopped at step 2750 and kept no state.
#   - the Inshrinkerator rule on seed 42 with the deletion counts of BO_s42_excp.
# Existing task ids are left unchanged, so the script can be run again.
set -eu
source /lihongliang/fangzl/dacp-a30/code/env.sh
Q=$CODE/queue_chain.tsv
[[ -e "$Q.bak_preS44" ]] || cp "$Q" "$Q.bak_preS44"
# the study B command of BO_s42_excp, with SEED in place of the seed (expanded by the worker at run time)
B='$PY $CODE/long_run.py --mode pretrain --tokenizer EleutherAI/pythia-410m --token-dir $DATA/fineweb_edu --total-steps 7500 --warmup 75 --lr 6e-4 --min-lr-ratio 0.1 --weight-decay 0.1 --beta2 0.95 --clip 1.0 --batch-size 4 --accum 16 --seq 2048 --eval-batches 40 --eval-batch-size 4 --ref-every 0 --stages 16 --score-batches 16 --score-batch-size 2 --amp bf16 --seed SEED --hidden 768 --layers 12 --heads 12 --ffn 3072 --ref-mode chain --ckpt-every 250 --break-every 1250 --eval-every 250 --eval-after 10,25,50,100,200 --save-every 250'
add() {
  if grep -q "^$1	" "$Q"; then echo "exists $1"; return; fi
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$Q"
  echo "added $1"
}
for s in 44 45; do
  b=${B//SEED/$s}
  add "BO_s${s}_excp" data/fineweb_edu/DONE "$b --method excp_official --excp-alpha 5e-5 --out \$RUNS/BO_s${s}_excp"
  add "BO_s${s}_dacpt" data/fineweb_edu/DONE,runs/BK_decide/DONE \
    "$b --method dacp --decision \$RUNS/BK_decide/decision.json --count-from \$RUNS/BO_s${s}_excp/metrics.jsonl --out \$RUNS/BO_s${s}_dacpt"
  add "BC_s${s}_nocomp" data/fineweb_edu/DONE "$b --method nocomp --out \$RUNS/BC_s${s}_nocomp"
done
b=${B//SEED/42}
add BO_s42_inshrink data/fineweb_edu/DONE,runs/BO_s42_excp/DONE \
  "$b --method inshrink --count-from \$RUNS/BO_s42_excp/metrics.jsonl --out \$RUNS/BO_s42_inshrink"
echo "tasks in queue: $(grep -vc '^#' "$Q")"
