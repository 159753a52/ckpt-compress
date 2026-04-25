#!/bin/bash
source /lihongliang/zm/miniconda3/etc/profile.d/conda.sh
conda activate GPT-2
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
unset http_proxy https_proxy
export HF_ENDPOINT=https://hf-mirror.com
python experiments/scripts/plot_weibull_fit.py 2>&1
