# 逐层剪枝对损失的影响分析

## 实验概述

本实验分析不同层次的参数剪枝对GPT-2模型损失的影响，使用一阶梯度绝对值 `|g·θ|` 作为重要性度量。

## 实验设计

### 模型和数据
- **模型**: GPT-2 Small (已在WikiText-103上微调1000步)
- **数据集**: WikiText-103
- **检查点**: `checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt`

### 重要性度量
- **公式**: `d_i = |g_i * θ_i|`
- **梯度累积**: 训练100步，累积梯度并取平均
- **优化器**: AdamW (lr=5e-5, weight_decay=0.01)

### 层次分组

实验对以下4个层次组进行剪枝分析：

1. **Embedding Layer** (2个参数)
   - `transformer.wte.weight`: Token Embedding [50257, 768]
   - `transformer.wpe.weight`: Position Embedding [1024, 768]

2. **Block 0 - Attention** (4个参数)
   - `transformer.h.0.attn.c_attn.weight`: QKV projection [768, 2304]
   - `transformer.h.0.attn.c_attn.bias`: QKV bias [2304]
   - `transformer.h.0.attn.c_proj.weight`: Output projection [768, 768]
   - `transformer.h.0.attn.c_proj.bias`: Output bias [768]

3. **Block 0 - MLP** (4个参数)
   - `transformer.h.0.mlp.c_fc.weight`: FC1 [768, 3072]
   - `transformer.h.0.mlp.c_fc.bias`: FC1 bias [3072]
   - `transformer.h.0.mlp.c_proj.weight`: FC2 [3072, 768]
   - `transformer.h.0.mlp.c_proj.bias`: FC2 bias [768]

4. **Block 0 - LayerNorm** (4个参数)
   - `transformer.h.0.ln_1.weight`: LayerNorm 1 [768]
   - `transformer.h.0.ln_1.bias`: LayerNorm 1 bias [768]
   - `transformer.h.0.ln_2.weight`: LayerNorm 2 [768]
   - `transformer.h.0.ln_2.bias`: LayerNorm 2 bias [768]

### 稀疏度设置

对每个层次组测试以下稀疏度：
- 5%, 10%, 15%, 20%, 25%, 30%

### 评估方法

1. **基线损失**: 未剪枝模型在10个固定批次上的平均损失
2. **剪枝评估**:
   - 对每个层次组和稀疏度组合
   - 根据重要性得分计算阈值
   - 将低于阈值的参数置零
   - 评估剪枝后的损失
   - 恢复原始权重（确保每次实验独立）
3. **损失增量**: `Δloss = loss_pruned - loss_baseline`

## 运行命令

```bash
python experiments/scripts/layer_pruning_analysis.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --batch_size 4 \
    --seq_length 512 \
    --num_eval_batches 10 \
    --device cuda \
    --output_dir results/layer_pruning_analysis
```

### 参数说明

- `--checkpoint`: 检查点路径
- `--num_steps`: 梯度累积步数（默认100）
- `--batch_size`: 批次大小（默认4）
- `--seq_length`: 序列长度（默认512）
- `--num_eval_batches`: 评估批次数（默认10）
- `--device`: 设备选择（cuda/cpu）
- `--output_dir`: 输出目录

## 输出结果

实验会生成以下文件：

### 1. CSV数据文件
`results/layer_pruning_analysis/pruning_results.csv`

包含以下列：
- `layer`: 层次组标识
- `layer_name`: 层次组名称
- `sparsity`: 稀疏度
- `loss`: 剪枝后的损失
- `baseline_loss`: 基线损失
- `loss_increase`: 损失增量
- `pruned_params`: 被剪枝的参数数量
- `total_params`: 该层次的总参数数量

### 2. 可视化图表

#### 图表1: Loss vs Sparsity
`results/layer_pruning_analysis/loss_vs_sparsity.png`

展示不同层次在不同稀疏度下的损失变化曲线。

#### 图表2: Loss Increase vs Sparsity
`results/layer_pruning_analysis/loss_increase_vs_sparsity.png`

展示不同层次在不同稀疏度下的损失增量（相对于基线）。

#### 图表3: Parameter Counts
`results/layer_pruning_analysis/parameter_counts.png`

展示各层次的参数数量统计。

## 预期结果分析

### 假设

1. **Embedding层**: 参数量大（~39M），但可能对损失影响较小
2. **Attention层**: 参数量中等（~2.4M），对模型性能关键
3. **MLP层**: 参数量大（~4.7M），可能有较大冗余
4. **LayerNorm层**: 参数量小（~3K），但可能对稳定性重要

### 关键问题

1. 哪个层次对损失最敏感？
2. 不同层次的剪枝容忍度如何？
3. 是否存在某些层次可以高度稀疏化而不影响性能？
4. 损失增量与稀疏度的关系是线性还是非线性？

## 实验流程

1. **加载阶段** (~5秒)
   - 加载模型检查点
   - 加载WikiText-103数据集
   - 准备评估批次

2. **基线评估** (~5秒)
   - 计算未剪枝模型的损失

3. **梯度累积** (~3-5分钟)
   - 训练100步
   - 累积梯度和Adam二阶矩
   - 计算重要性得分

4. **剪枝实验** (~2-3分钟)
   - 4个层次组 × 6个稀疏度 = 24次实验
   - 每次实验：剪枝 → 评估 → 恢复

5. **结果保存** (~1秒)
   - 保存CSV数据
   - 生成可视化图表

**总耗时**: 约5-10分钟

## 技术细节

### 剪枝策略

```python
# 1. 计算阈值
threshold = torch.quantile(flat_scores, sparsity)

# 2. 创建掩码
mask = (scores >= threshold).float()

# 3. 应用剪枝
param.data.mul_(mask)

# 4. 评估后恢复
param.data.copy_(original_weights)
```

### 内存管理

- 模型在GPU上运行
- 梯度和重要性得分在CPU上累积
- 每次剪枝实验独立，不累积内存

### 数据一致性

- 使用固定的10个评估批次
- 确保所有实验使用相同的数据
- 避免数据随机性影响结果

## 参考脚本

本实验参考了以下脚本的实现：
- `analyze_all_blocks_importance_first_order.py`: 梯度累积和重要性计算
- `src/ckpt_compress/methods/adam_prune/importance.py`: 重要性得分计算函数

## 后续扩展

可能的扩展方向：

1. **更多层次**: 分析所有12个Transformer blocks
2. **更多稀疏度**: 测试更细粒度的稀疏度（1%-50%）
3. **不同重要性度量**: 对比一阶、二阶、幅值等方法
4. **联合剪枝**: 同时剪枝多个层次
5. **微调恢复**: 剪枝后进行微调，观察损失恢复情况
