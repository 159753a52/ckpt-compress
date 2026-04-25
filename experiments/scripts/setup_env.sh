#!/bin/bash
# ============================================================
# 远程服务器环境配置脚本
# 用途：配置 HF 缓存、数据路径、符号链接，使论文实验可离线运行
# 使用：source experiments/scripts/setup_env.sh
# ============================================================

set -e

# ---------- 网络/代理 ----------
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
export HF_ENDPOINT="https://hf-mirror.com"

# ---------- HuggingFace 离线模式 ----------
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ---------- 缓存目录 ----------
# Pythia-410M 和 WikiText-103 缓存在 fangzl 的 HF cache 中
export HF_HOME="/lihongliang/fangzl/.cache/huggingface"
export HF_DATASETS_CACHE="/root/.cache/huggingface/datasets"

# GPT-2/BERT 等模型同时在 /root/.cache 和 fangzl cache 中
# transformers 会依次搜索 HF_HOME/hub/ 下的模型

# ---------- 符号链接：合并两处模型缓存 ----------
# 将 /root 的 HF 模型缓存链接到 fangzl 缓存目录
# 这样 from_pretrained 能同时找到两处的模型
_merge_hf_cache() {
    local src="/root/.cache/huggingface/hub"
    local dst="/lihongliang/fangzl/.cache/huggingface/hub"
    
    if [ ! -d "$src" ] || [ ! -d "$dst" ]; then
        echo "[WARN] 缓存目录不存在，跳过合并"
        return
    fi
    
    for model_dir in "$src"/models--*; do
        [ -d "$model_dir" ] || continue
        local name=$(basename "$model_dir")
        if [ ! -e "$dst/$name" ]; then
            ln -sf "$model_dir" "$dst/$name"
            echo "[LINK] $name -> $dst/$name"
        fi
    done
}
_merge_hf_cache

# ---------- 符号链接：WikiText-103 数据集 ----------
# fangzl cache 中有 WikiText-103 arrow 文件，链接到 root datasets cache
_link_wikitext103() {
    local src="/lihongliang/fangzl/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1"
    local dst="/root/.cache/huggingface/datasets/wikitext/wikitext-103-raw-v1"
    
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        mkdir -p "$(dirname $dst)"
        ln -sf "$src" "$dst"
        echo "[LINK] WikiText-103 dataset cache linked"
    fi
}
_link_wikitext103

# ---------- ViT-L/32 模型映射 ----------
# 远程只有 vit-large-patch32-384，需要：
#   1. 在 HF cache 中创建 224 -> 384 的符号链接（模型代码会加载 384 版本）
#   2. 同时修改 models.py 中的 HF 名称（见 patch_vit_model_name）
# 方案：直接创建同名链接让代码透明加载
_link_vit() {
    local src_384="/lihongliang/fangzl/ckpt-compress/data/models/models--google--vit-large-patch32-384"
    local hub_dir="/lihongliang/fangzl/.cache/huggingface/hub"
    local dst_link="$hub_dir/models--google--vit-large-patch32-384"
    
    if [ -d "$src_384" ] && [ ! -e "$dst_link" ]; then
        ln -sf "$src_384" "$dst_link"
        echo "[LINK] ViT-L/32-384 linked to HF hub cache"
    fi
}
_link_vit

# ---------- ImageNet 数据路径 ----------
# bobzhou 目录下有完整 ImageNet（146GB）
# 代码通过 data_dir 参数找 {data_dir}/imagenet/train 和 val
export IMAGENET_DATA_DIR="/lihongliang/bobzhou/dataset"

# ---------- Alpaca 数据路径 ----------
# 本地有 alpaca_data.json，代码需要从 HF load_dataset 加载
# 方案：设置环境变量供脚本识别
export ALPACA_LOCAL_PATH="/lihongliang/fangzl/ckpt-compress/data/alpaca/alpaca_data_fixed.json"

# ---------- Checkpoint 目录 ----------
export CHECKPOINT_BASE="/root/checkpoints"
mkdir -p "$CHECKPOINT_BASE"

# ---------- Conda 环境 ----------
CONDA_ENV="/lihongliang/zm/miniconda3/envs/GPT-2"
if [ -d "$CONDA_ENV" ]; then
    export PATH="$CONDA_ENV/bin:$PATH"
    echo "[ENV] Using conda env: $CONDA_ENV"
fi

# ---------- GPU 检查 ----------
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "[GPU] $GPU_NAME ($GPU_MEM)"
fi

# ---------- 工作目录 ----------
WORK_DIR="/lihongliang/fangzl/ckpt-compress/code/checkpoint_compress"
if [ -d "$WORK_DIR" ]; then
    cd "$WORK_DIR"
    echo "[CWD] $(pwd)"
fi

# ---------- 汇总 ----------
echo ""
echo "============================================"
echo "  环境配置完成"
echo "============================================"
echo "  HF_HOME:              $HF_HOME"
echo "  HF_DATASETS_CACHE:    $HF_DATASETS_CACHE"
echo "  TRANSFORMERS_OFFLINE:  $TRANSFORMERS_OFFLINE"
echo "  HF_DATASETS_OFFLINE:  $HF_DATASETS_OFFLINE"
echo "  IMAGENET_DATA_DIR:    $IMAGENET_DATA_DIR"
echo "  CHECKPOINT_BASE:      $CHECKPOINT_BASE"
echo "  工作目录:             $(pwd)"
echo "============================================"
echo ""
echo "可用模型（HF cache）:"
ls "$HF_HOME/hub/" 2>/dev/null | grep "^models--" | sed 's/models--/  /' || echo "  (无)"
echo ""
echo "可用数据集（HF cache）:"
ls "$HF_HOME/datasets/" 2>/dev/null | grep -v lock | grep -v downloads || echo "  (无)"
echo ""
echo "ImageNet: $([ -d $IMAGENET_DATA_DIR/imagenet/train ] && echo '✅ 可用' || echo '❌ 不可用')"
echo "Alpaca:   $([ -f $ALPACA_LOCAL_PATH ] && echo '✅ 可用' || echo '❌ 不可用')"
