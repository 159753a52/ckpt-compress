# 所有Blocks无损剪枝分析脚本使用说明

## 脚本功能

`all_blocks_pruning_analysis.py` 用于测试GPT-2 Small所有12个Transformer Blocks中每个参数组的无损剪枝比例。

### 测试范围

- **Embedding Layer**: Token Embedding + Position Embedding
- **12 个 Transformer Blocks**，每个block包含：
  - Attention Layer
  - MLP Layer
  - LayerNorm Layer

### 测试稀疏度

5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 55%, 60%

### 无损标准

损失增量 ≤ 0.001（即不增加损失或损失增加小于0.001）

---

## 运行命令

### 基本用法

```bash
python experiments/scripts/all_blocks_pruning_analysis.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --batch_size 4 \
    --seq_length 512 \
    --num_eval_batches 10 \
    --device cuda \
    --output_dir results/all_blocks_pruning_analysis
```

### 参数说明

- `--checkpoint`: 检查点路径（必需）
- `--num_steps`: 梯度累积步数（默认100）
- `--batch_size`: 批次大小（默认4）
- `--seq_length`: 序列长度（默认512）
- `--num_eval_batches`: 评估批次数（默认10）
- `--device`: 设备选择 cuda/cpu（默认cuda）
- `--output_dir`: 输出目录（默认results/all_blocks_pruning_analysis）

---

## 输出文件

### 1. `all_blocks_pruning_results.csv`
完整的实验数据，包含：
- layer: 层次类型（embedding/attn/mlp/ln）
- block_id: Block编号（-1表示Embedding层，0-11表示Transformer Blocks）
- sparsity: 稀疏度
- loss: 剪枝后的损失
- loss_increase: 损失增量
- pruned_params: 剪枝的参数数量
- total_params: 总参数数量

### 2. `lossless_sparsity_summary.csv`
无损剪枝摘要，包含每个层次的最大无损剪枝比例：
- layer: 层次类型
- block_id: Block编号
- max_lossless_sparsity: 最大无损稀疏度（0-1）
- max_lossless_percent: 最大无损稀疏度百分比

### 3. `lossless_sparsity_heatmap.png`
热力图，展示每个Block的每个层次的最大无损剪枝比例

### 4. `block_comparison.png`
对比图，展示不同Block在不同稀疏度下的损失增量曲线

### 5. `ANALYSIS_REPORT.md`
详细的分析报告，包含：
- 每个层次的统计信息（平均、标准差、最小、最大）
- 最耐剪枝和最敏感的Block
- 关键发现

---

## 预期运行时间

- **梯度累积**: ~5-10分钟（100步）
- **剪枝实验**: ~2-3小时
  - Embedding: 12个稀疏度测试
  - 12 Blocks × 3 层次 × 12 稀疏度 = 432次测试
- **总计**: ~2.5-3.5小时

---

## 内存需求

- **GPU显存**: 建议16GB+（用于GPT-2 Small）
- **系统内存**: 建议32GB+

---

## 示例输出

### 无损剪枝摘要示例

```
| Layer      | Block ID | Max Lossless Sparsity (%) |
|------------|----------|---------------------------|
| Embedding  | -1       | 5.0%                      |
| Attention  | 0        | 30.0%                     |
| MLP        | 0        | 60.0%                     |
| LayerNorm  | 0        | 50.0%                     |
| Attention  | 1        | 35.0%                     |
| MLP        | 1        | 55.0%                     |
| ...        | ...      | ...                       |
```

### 关键发现示例

```
## 关键发现

### Attention
- 最耐剪枝: Block 5 (35.0%)
- 最敏感: Block 0 (30.0%)

### MLP
- 最耐剪枝: Block 3 (60.0%)
- 最敏感: Block 11 (50.0%)

### LayerNorm
- 最耐剪枝: Block 2 (55.0%)
- 最敏感: Block 10 (45.0%)
```

---

## 注意事项

1. **实验时间较长**: 完整实验需要2-3小时，建议在后台运行
2. **检查点要求**: 需要使用微调后的检查点（包含model_state_dict）
3. **数据集**: 自动从HuggingFace下载WikiText-103
4. **GPU推荐**: 建议使用GPU加速，CPU运行会非常慢

---

## 快速测试

如果想快速测试脚本是否正常工作，可以减少测试范围：

```bash
# 修改脚本中的稀疏度列表（第485行）
# sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 只测试6个稀疏度

# 或者只测试前3个blocks
# for block_id in range(3):  # 修改第520行
```

---

## 故障排除

### 1. CUDA Out of Memory
- 减小 `--batch_size`（如改为2）
- 减小 `--num_eval_batches`（如改为5）

### 2. 数据加载失败
- 检查网络连接（需要下载WikiText-103）
- 或手动下载数据集到 `./data/` 目录

### 3. 检查点加载失败
- 确认检查点路径正确
- 确认检查点包含 `model_state_dict` 键

---

## 后续分析

实验完成后，可以：

1. 查看 `lossless_sparsity_summary.csv` 了解每个层次的无损剪枝比例
2. 查看 `lossless_sparsity_heatmap.png` 直观了解不同Block的差异
3. 查看 `ANALYSIS_REPORT.md` 获取详细分析
4. 基于结果设计联合剪枝策略

---

## 相关脚本

- `layer_pruning_analysis.py`: 测试Block 0的低稀疏度（5%-30%）
- `layer_pruning_analysis_high_sparsity.py`: 测试Block 0的高稀疏度（40%-90%）
- `all_blocks_pruning_analysis.py`: 测试所有Blocks（本脚本）
