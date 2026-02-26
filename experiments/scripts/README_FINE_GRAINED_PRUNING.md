# 细粒度剪枝分析脚本使用说明

## 脚本功能

`fine_grained_pruning_analysis.py` 用于测试 GPT-2 Small 所有 12 个 Transformer Blocks 中每个 **weight 参数**的无损剪枝比例。

### 与 `all_blocks_pruning_analysis.py` 的区别

| 特性 | all_blocks_pruning_analysis.py | fine_grained_pruning_analysis.py |
|------|-------------------------------|----------------------------------|
| **粒度** | 粗粒度（attn/mlp/ln） | 细粒度（6个子层） |
| **Attention** | 整个 Attention 层 | attn_qkv + attn_proj |
| **MLP** | 整个 MLP 层 | mlp_fc + mlp_proj |
| **LayerNorm** | 整个 LayerNorm 层 | ln_1 + ln_2 |
| **参数类型** | weight + bias | **只测试 weight** |

---

## 测试范围

### Embedding Layer
- Token Embedding (wte.weight)
- Position Embedding (wpe.weight)

### 12 个 Transformer Blocks，每个包含 6 个细粒度层次：

#### Attention 层（2个子层）
- **attn_qkv**: `c_attn.weight` [768, 2304] - QKV 投影矩阵
- **attn_proj**: `c_proj.weight` [768, 768] - 输出投影矩阵

#### MLP 层（2个子层）
- **mlp_fc**: `c_fc.weight` [768, 3072] - 第一层（扩展层）
- **mlp_proj**: `c_proj.weight` [3072, 768] - 第二层（投影层）

#### LayerNorm 层（2个子层）
- **ln_1**: `ln_1.weight` [768] - Attention 前的 LayerNorm
- **ln_2**: `ln_2.weight` [768] - MLP 前的 LayerNorm

---

## 测试稀疏度

5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%, 55%, 60%, 65%, 70%, 75%, 80%, 85%, 90%

---

## 无损标准

损失增量 ≤ 0.001（即不增加损失或损失增加小于 0.001）

---

## 运行命令

### 基本用法

```bash
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --batch_size 4 \
    --seq_length 512 \
    --num_eval_batches 10 \
    --device cuda \
    --output_dir results/fine_grained_pruning_analysis
```

### 参数说明

- `--checkpoint`: 检查点路径（必需）
- `--num_steps`: 梯度累积步数（默认100）
- `--batch_size`: 批次大小（默认4）
- `--seq_length`: 序列长度（默认512）
- `--num_eval_batches`: 评估批次数（默认10）
- `--device`: 设备选择 cuda/cpu（默认cuda）
- `--output_dir`: 输出目录（默认results/fine_grained_pruning_analysis）

---

## 输出文件

### 1. `fine_grained_pruning_results.csv`
完整的实验数据，包含：
- layer: 层次类型（embedding/attn_qkv/attn_proj/mlp_fc/mlp_proj/ln_1/ln_2）
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
热力图，展示每个Block的每个细粒度层次的最大无损剪枝比例

### 4. `block_comparison.png`
对比图（2×3布局），展示不同Block在不同稀疏度下的损失增量曲线：
- Attention QKV
- Attention Proj
- MLP FC
- MLP Proj
- LayerNorm 1
- LayerNorm 2

### 5. `ANALYSIS_REPORT.md`
详细的分析报告，包含：
- 每个层次的统计信息（平均、标准差、最小、最大）
- 最耐剪枝和最敏感的Block
- 层次对比分析（QKV vs Proj, FC vs Proj）
- 关键发现

---

## 预期运行时间

- **梯度累积**: ~5-10 分钟（100步）
- **剪枝实验**: ~5-6 小时
  - Embedding: 18个稀疏度测试
  - 12 Blocks × 6 层次 × 18 稀疏度 = 1296次测试
- **总计**: ~5.5-6.5 小时

---

## 内存需求

- **GPU显存**: 建议16GB+（用于GPT-2 Small）
- **系统内存**: 建议32GB+

---

## 示例输出

### 无损剪枝摘要示例

```
| Layer       | Block ID | Max Lossless Sparsity (%) |
|-------------|----------|---------------------------|
| Embedding   | -1       | 5.0%                      |
| Attn QKV    | 0        | 25.0%                     |
| Attn Proj   | 0        | 35.0%                     |
| MLP FC      | 0        | 55.0%                     |
| MLP Proj    | 0        | 60.0%                     |
| LayerNorm 1 | 0        | 45.0%                     |
| LayerNorm 2 | 0        | 50.0%                     |
| ...         | ...      | ...                       |
```

### 层次对比分析示例

```
## 层次对比分析

### Attention 层对比
- **QKV 平均无损剪枝**: 28.3%
- **Proj 平均无损剪枝**: 36.7%

### MLP 层对比
- **FC 平均无损剪枝**: 52.1%
- **Proj 平均无损剪枝**: 58.9%
```

---

## 关键发现（预期）

基于细粒度分析，你可能会发现：

1. **Attention 层内部差异**：
   - QKV 投影矩阵可能比输出投影矩阵更敏感
   - 或者相反，取决于模型训练状态

2. **MLP 层内部差异**：
   - 第一层（扩展层）vs 第二层（投影层）的剪枝容忍度
   - 可能发现某一层更适合剪枝

3. **LayerNorm 差异**：
   - Attention 前的 LN vs MLP 前的 LN
   - 可能有不同的剪枝敏感度

4. **跨 Block 模式**：
   - 浅层 vs 深层的差异
   - 中间层可能更耐剪枝

---

## 注意事项

1. **实验时间较长**: 完整实验需要3-4小时，建议在后台运行
2. **检查点要求**: 需要使用微调后的检查点（包含model_state_dict）
3. **数据集**: 自动从HuggingFace下载WikiText-103
4. **GPU推荐**: 建议使用GPU加速，CPU运行会非常慢
5. **只测试 weight**: 排除所有 bias 参数

---

## 快速测试

如果想快速测试脚本是否正常工作，可以减少测试范围：

```bash
# 方法1: 修改脚本中的稀疏度列表（第577行）
# sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]  # 只测试10个稀疏度

# 方法2: 只测试前3个blocks（修改第632行）
# for block_id in range(3):

# 方法3: 只测试部分层次（修改第620-627行）
# layer_configs = [
#     ('attn_qkv', 'Attention QKV'),
#     ('mlp_fc', 'MLP FC'),
# ]
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

### 4. 参数未找到警告
- 检查模型结构是否与预期一致
- 确认使用的是 GPT-2 Small 模型

---

## 后续分析

实验完成后，可以：

1. **查看 CSV 文件**：
   - `lossless_sparsity_summary.csv` - 了解每个细粒度层次的无损剪枝比例
   - `fine_grained_pruning_results.csv` - 完整的实验数据

2. **查看可视化**：
   - `lossless_sparsity_heatmap.png` - 直观了解不同Block和层次的差异
   - `block_comparison.png` - 对比不同层次的剪枝敏感度曲线

3. **查看分析报告**：
   - `ANALYSIS_REPORT.md` - 详细的统计分析和关键发现

4. **设计剪枝策略**：
   - 基于结果设计非均匀剪枝策略
   - 对不同层次应用不同的剪枝比例
   - 优先剪枝耐剪枝的层次

---

## 与其他脚本的对比

| 脚本 | 粒度 | 测试范围 | 参数类型 | 实验次数 |
|------|------|----------|----------|----------|
| `layer_pruning_analysis.py` | 粗粒度 | Block 0 | weight + bias | 36次 |
| `all_blocks_pruning_analysis.py` | 粗粒度 | 所有Blocks | weight + bias | 444次 |
| `fine_grained_pruning_analysis.py` | **细粒度** | **所有Blocks** | **只 weight** | **1314次** |

---

## 相关脚本

- `layer_pruning_analysis.py`: 测试Block 0的低稀疏度（5%-30%）
- `layer_pruning_analysis_high_sparsity.py`: 测试Block 0的高稀疏度（40%-90%）
- `all_blocks_pruning_analysis.py`: 测试所有Blocks（粗粒度）
- `fine_grained_pruning_analysis.py`: 测试所有Blocks（细粒度，本脚本）

---

## 使用建议

1. **先运行粗粒度分析**：使用 `all_blocks_pruning_analysis.py` 快速了解整体情况
2. **再运行细粒度分析**：使用本脚本深入了解每个子层的特性
3. **对比分析**：比较粗粒度和细粒度的结果，发现内部差异
4. **设计策略**：基于细粒度结果设计更精细的剪枝策略

---

## 预期收益

通过细粒度分析，你可以：

1. **发现内部差异**：了解 Attention 和 MLP 内部不同子层的剪枝特性
2. **优化剪枝策略**：对不同子层应用不同的剪枝比例
3. **提高压缩率**：在保持性能的前提下，最大化剪枝比例
4. **理解模型结构**：深入理解 Transformer 不同组件的重要性

---

## 示例工作流

```bash
# 1. 运行细粒度剪枝分析
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --device cuda \
    --output_dir results/fine_grained_pruning_analysis

# 2. 查看结果
cd results/fine_grained_pruning_analysis
cat ANALYSIS_REPORT.md
open lossless_sparsity_heatmap.png
open block_comparison.png

# 3. 分析 CSV 数据
python -c "
import pandas as pd
df = pd.read_csv('lossless_sparsity_summary.csv')
print(df.groupby('layer')['max_lossless_percent'].describe())
"

# 4. 基于结果设计剪枝策略
# 例如：对 MLP Proj 应用 60% 剪枝，对 Attn QKV 应用 25% 剪枝
```
