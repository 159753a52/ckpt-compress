# 实验指南

本文档提供 `ckpt-compress` 项目中各种实验脚本的详细使用说明。

## 目录

- [数据准备](#数据准备)
- [训练脚本](#训练脚本)
- [检查点压缩](#检查点压缩)
- [AdamPrune 实验](#adamprune-实验)
- [压缩方法对比实验](#压缩方法对比实验)
- [可视化和分析脚本](#可视化和分析脚本)
- [配置文件使用](#配置文件使用)
- [实验配置建议](#实验配置建议)

---

## 数据准备

在运行实验前，请先下载所需的数据集和模型。详见 [DATA_PREPARATION.md](DATA_PREPARATION.md)。

```bash
# 快速下载所有数据和模型
python scripts/download_data.py --all
python scripts/download_models.py --all
```

---

## 训练脚本

### CV 模型训练

#### ResNet18 on CIFAR-10

```bash
python experiments/scripts/train_cv.py \
  --model resnet18 \
  --dataset cifar10 \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1 \
  --lr_schedule cosine \
  --checkpoint_dir ./checkpoints/resnet18_cifar10
```

#### ResNet50 on CIFAR-100

```bash
python experiments/scripts/train_cv.py \
  --model resnet50 \
  --dataset cifar100 \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1 \
  --lr_schedule cosine \
  --checkpoint_dir ./checkpoints/resnet50_cifar100
```

### NLP 模型训练

#### GPT-2 Small on WikiText-2

```bash
python experiments/scripts/train_nlp.py \
  --model gpt2-small \
  --dataset wikitext2 \
  --epochs 10 \
  --batch_size 8 \
  --lr 5e-5 \
  --seq_length 512 \
  --checkpoint_dir ./checkpoints/gpt2_small_wikitext2
```

#### GPT-2 Medium on WikiText-103

```bash
python experiments/scripts/train_nlp.py \
  --model gpt2-medium \
  --dataset wikitext103 \
  --epochs 10 \
  --batch_size 4 \
  --lr 3e-5 \
  --seq_length 512 \
  --use_amp \
  --gradient_accumulation_steps 8 \
  --checkpoint_dir ./checkpoints/gpt2_medium_wikitext103
```

**注意**: GPT-2 Medium 训练需要大量内存，建议使用 `--use_amp` 和 `--gradient_accumulation_steps`。

---

## 检查点压缩

### 压缩检查点

支持三种压缩方法：`excp`、`inshrinkerator`、`predictive`

```bash
# 使用 ExCP 方法压缩
python experiments/scripts/compress_and_resume.py \
  --mode compress \
  --checkpoint checkpoints/resnet18_cifar10/checkpoint_epoch_100.pt \
  --method excp \
  --output compressed/resnet18_epoch100_excp.bin
```

### 解压检查点

```bash
python experiments/scripts/compress_and_resume.py \
  --mode decompress \
  --compressed compressed/resnet18_epoch100_excp.bin \
  --output checkpoints/resnet18_epoch100_restored.pt
```

---

## 配置文件使用

项目提供了预定义的 YAML 配置文件，位于 `experiments/configs/` 目录。

### 可用配置文件

**CV 配置**:
- `cv/resnet18_cifar10.yaml`
- `cv/resnet18_cifar100.yaml`
- `cv/resnet50_cifar100.yaml`

**NLP 配置**:
- `nlp/gpt2_small_wikitext2.yaml`
- `nlp/gpt2_small_wikitext103.yaml`
- `nlp/gpt2_medium_wikitext103.yaml`

### 使用配置文件

虽然训练脚本目前主要使用命令行参数，但配置文件提供了推荐的超参数设置，可以作为参考。

---

## AdamPrune 实验

### 1. 公式对比实验 (`run_formula_comparison.py`)

**功能**: 对比 Adam 近似和 HVP 精确两种重要性得分计算方法。

**基本用法**:
```bash
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 100
```

**主要参数**:
- `--mode`: 实验模式
  - `comparison`: 公式对比模式
  - `calibration`: 幂律校正模式
- `--max_samples`: 限制数据集大小（推荐 100-500）
- `--num_batches`: 缓存的批次数（默认 10）
- `--batch_size`: 每批次大小（默认 4）
- `--seq_length`: 序列长度（默认 128）
- `--local_path`: 本地 WikiText-2 数据路径
- `--output_dir`: 输出目录（默认 `results/formula_comparison_<timestamp>/`）

**示例**:

```bash
# 使用本地数据运行公式对比
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 100 \
  --local_path data/wikitext-2

# 运行幂律校正实验
python experiments/scripts/run_formula_comparison.py \
  --mode calibration \
  --max_samples 100 \
  --sparsities 0.005,0.01,0.02,0.04,0.06,0.08
```

**输出**:
- `experiment_report.md`: 实验报告
- `layer_distributions.png`: 层重要性分布图
- `marginal_loss_curves.png`: 边际损失曲线
- `calibration_scatter.png`: 校正散点图（仅 calibration 模式）

### 2. 自适应剪枝实验 (`run_adaptive_pruning.py`)

**功能**: 分布拟合 + 幂律校正 + 最优稀疏度搜索。

**基本用法**:
```bash
python experiments/scripts/run_adaptive_pruning.py \
  --max_samples 100
```

**主要参数**:
- `--max_samples`: 限制数据集大小
- `--num_batches`: 缓存的批次数
- `--target_loss`: 目标损失容忍度（默认 0.01）
- `--output_dir`: 输出目录

**输出**:
- `layer_prune_ratios.csv`: 每层剪枝比例
- `layer_type_summary.csv`: 按层类型汇总
- `calibration_scatter.png`: 校正散点图
- `sparsity_vs_loss.png`: 稀疏度 vs 损失曲线
- `report.json`: 完整实验报告

### 3. 验证脚本 (`verify_adam_prune.py`)

**功能**: 验证 AdamPrune 方法的正确性。

**基本用法**:
```bash
python experiments/scripts/verify_adam_prune.py
```

## 压缩方法对比实验

### 压缩方法对比 (`run_comparison.py`)

**功能**: 对比 ExCP vs Inshrinkerator vs PredictiveResidual 在 ResNet18 + CIFAR-10 上的性能。

**基本用法**:
```bash
python experiments/scripts/run_comparison.py
```

**实验流程**:
1. 在 CIFAR-10 上训练 ResNet18 若干轮
2. 定期保存检查点
3. 使用三种方法压缩检查点
4. 解压并验证重建
5. 从重建的检查点恢复训练
6. 报告压缩率和训练稳定性

**输出**:
- 压缩率对比
- 重建误差
- 训练恢复性能

## 可视化和分析脚本

### 1. 可视化重要性分布 (`visualize_importance.py`)

**功能**: 可视化参数重要性分布。

**基本用法**:
```bash
python experiments/scripts/visualize_importance.py
```

### 2. 分析 GPT-2 重要性分布 (`analyze_gpt2_distribution.py`)

**功能**: 分析 GPT-2 模型的重要性分布特征。

**基本用法**:
```bash
python experiments/scripts/analyze_gpt2_distribution.py
```

### 3. 分析 GPT-2 各层分布 (`analyze_gpt2_layer_distributions.py`)

**功能**: 分析 GPT-2 各层的重要性分布。

**基本用法**:
```bash
python experiments/scripts/analyze_gpt2_layer_distributions.py
```

### 4. 可视化 GPT-2 分布（多种版本）

**功能**: 提供多种可视化方式。

**基本用法**:
```bash
# 标准版本
python experiments/scripts/visualize_gpt2_distributions.py

# 所有层版本
python experiments/scripts/visualize_gpt2_all_layers.py

# 简化版本
python experiments/scripts/visualize_gpt2_simple.py

# 修正版本
python experiments/scripts/visualize_gpt2_fixed.py
```

### 5. 分析分布类型 (`analyze_distribution_type.py`)

**功能**: 分析重要性分布的类型（韦伯、指数等）。

**基本用法**:
```bash
python experiments/scripts/analyze_distribution_type.py
```

### 6. 生成数据文件

**CSV 数据**:
```bash
python experiments/scripts/generate_csv_data.py
```

**块重要性 TSV 数据**:
```bash
python experiments/scripts/generate_block_importance_tsv.py
```

## 实验配置建议

### GPT-2 实验（内存敏感）

**推荐配置**:
```bash
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 100 \
  --num_batches 10 \
  --batch_size 4 \
  --seq_length 128 \
  --memory_limit_gb 30
```

**内存优化策略**:
- 使用 `--max_samples` 限制数据集大小（100-500）
- 减少 `--num_batches`（5-10）
- 减少 `--batch_size`（2-4）
- 减少 `--seq_length`（64-128）
- 设置 `--memory_limit_gb`（20-30）

**硬件要求**:
- **最低**: 16GB RAM, 8GB VRAM
- **推荐**: 32GB RAM, 16GB VRAM
- **理想**: 64GB RAM, 24GB VRAM

### CIFAR-10 实验（轻量级）

**推荐配置**:
```bash
python experiments/scripts/run_comparison.py
```

**硬件要求**:
- **最低**: 8GB RAM, 4GB VRAM
- **推荐**: 16GB RAM, 8GB VRAM

### 数据路径配置

**使用本地 WikiText-2 数据**:
```bash
# 下载数据（首次）
mkdir -p data/wikitext-2
# 手动下载或使用 HuggingFace datasets 库

# 使用本地数据
python experiments/scripts/run_formula_comparison.py \
  --local_path data/wikitext-2
```

**HuggingFace 缓存配置**:
```python
from ckpt_compress.methods.adam_prune.adaptive_pruning import setup_hf_cache

# 设置缓存目录
setup_hf_cache("data/hf_cache")
```

### 实验结果管理

**结果目录结构**:
```
results/
├── calibration_<timestamp>/
│   ├── experiment_report.md
│   ├── calibration_scatter.png
│   └── ...
├── formula_comparison_<timestamp>/
│   ├── experiment_report.md
│   ├── layer_distributions.png
│   └── ...
└── adaptive_pruning_<timestamp>/
    ├── layer_prune_ratios.csv
    ├── report.json
    └── ...
```

**清理旧结果**:
```bash
# 删除超过 30 天的结果
find results/ -type d -mtime +30 -exec rm -rf {} +
```

## 常见问题

### 1. 内存不足（OOM）

**症状**: `RuntimeError: CUDA out of memory` 或进程被 killed

**解决方案**:
- 减少 `--max_samples`
- 减少 `--batch_size`
- 减少 `--seq_length`
- 使用 CPU 卸载（在代码中设置 `to_cpu=True`）

### 2. 分布拟合失败

**症状**: 警告 `Distribution fitting failed for layer X`

**原因**: 某些层的重要性分布不符合预设的分布类型

**解决方案**:
- 检查该层的重要性分布
- 尝试其他分布类型
- 使用全局稀疏度剪枝代替分层剪枝

### 3. 实验运行缓慢

**原因**:
- 数据集太大
- HVP 计算开销大
- 分布拟合计算密集

**解决方案**:
- 使用 `--max_samples` 限制数据集大小
- 使用 Adam 近似代替 HVP
- 使用 `fit_distributions_per_layer_fast()` 加速分布拟合

### 4. WikiText-2 下载失败

**症状**: `ConnectionError` 或 `TimeoutError`

**解决方案**:
- 使用本地数据：`--local_path data/wikitext-2`
- 配置代理
- 手动下载数据集

## 实验最佳实践

### 1. 逐步增加复杂度

```bash
# 第一步：小规模验证
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 50

# 第二步：中等规模
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 200

# 第三步：完整实验
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 1000
```

### 2. 监控内存使用

```bash
# 使用 watch 监控内存
watch -n 1 'nvidia-smi; free -h'

# 在脚本中启用内存监控
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 100 \
  --memory_limit_gb 30  # 启用内存限制
```

### 3. 保存中间结果

实验脚本会自动保存中间结果，如果实验中断可以从中间结果恢复。

### 4. 并行运行实验

```bash
# 使用不同配置并行运行
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 100 \
  --output_dir results/exp1 &

python experiments/scripts/run_formula_comparison.py \
  --mode calibration \
  --max_samples 100 \
  --output_dir results/exp2 &
```

### 5. 记录实验配置

在实验报告中会自动记录配置参数，建议额外保存实验命令：

```bash
# 保存实验命令
echo "python experiments/scripts/run_formula_comparison.py --mode comparison --max_samples 100" \
  > results/formula_comparison_<timestamp>/command.txt
```

## 高级用法

### 自定义重要性得分公式

修改 `src/ckpt_compress/methods/adam_prune/importance.py` 中的 `compute_importance_scores()` 函数。

### 自定义分布类型

修改 `src/ckpt_compress/methods/adam_prune/distribution.py` 中的 `fit_multiple_distributions()` 函数。

### 自定义剪枝策略

修改 `src/ckpt_compress/methods/adam_prune/layer_pruning.py` 中的 `LayerPruningOptimizer` 类。

## 参考资料

- [ARCHITECTURE.md](ARCHITECTURE.md) - 详细架构设计
- [CLAUDE.md](../CLAUDE.md) - 快速参考手册
