#!/bin/bash
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress

DATA_DIR=/root/data
CKPT_DIR=/root/checkpoints/bert_large_sst2_1000steps
mkdir -p $DATA_DIR $CKPT_DIR

echo "========== Step 1: BERT-Large SST2 Finetune =========="
python experiments/scripts/finetune/finetune_bert_large.py --dataset sst2 --num_steps 1000 --batch_size 16 --lr 2e-5 --gradient_accumulation_steps 2 --max_length 128 --save_interval 200 --data_dir $DATA_DIR --checkpoint_dir $CKPT_DIR --device cuda --use_amp

echo ""
echo "========== Step 2: Gamma Validation =========="
python experiments/scripts/run_gamma_validation.py --model bert-large --dataset sst2 --checkpoint $CKPT_DIR/checkpoint_step_1000.pt --prune_ratios 0.1,0.2,0.3,0.4 --num_steps 50 --eval_batches 20 --hvp_batches 1 --seq_length 128 --batch_size 8 --hvp_mode block --data_dir $DATA_DIR --device cuda

echo "========== ALL DONE =========="