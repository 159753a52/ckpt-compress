# 剪枝分析工具总结

## 📊 你现在拥有的工具

### 1. **细粒度剪枝分析脚本** ✨ NEW
**文件**: `experiments/scripts/fine_grained_pruning_analysis.py`

**特点**:
- ✅ **只测试 weight 参数**（排除 bias）
- ✅ **细粒度层次划分**（6个子层）
- ✅ 可以发现 Attention 和 MLP 内部的差异

**层次划分**:
```
Attention (2个子层):
  - attn_qkv: c_attn.weight [768, 2304] - QKV 投影矩阵
  - attn_proj: c_proj.weight [768, 768] - 输出投影矩阵

MLP (2个子层):
  - mlp_fc: c_fc.weight [768, 3072] - 第一层（扩展层）
  - mlp_proj: c_proj.weight [3072, 768] - 第二层（投影层）

LayerNorm (2个子层):
  - ln_1: ln_1.weight [768] - Attention 前
  - ln_2: ln_2.weight [768] - MLP 前
```

**运行命令**:
```bash
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --batch_size 4 \
    --device cuda \
    --output_dir results/fine_grained_pruning_analysis
```

**实验次数**: 1 (Embedding) + 12 blocks × 6 layers × 12 sparsities = **876 次**

**预期时间**: ~3.5-4.5 小时

---

### 2. **粗粒度剪枝分析脚本**
**文件**: `experiments/scripts/all_blocks_pruning_analysis.py`

**特点**:
- 包含 weight + bias
- 粗粒度层次划分（3个层次）
- 快速了解整体情况

**层次划分**:
```
- attn: 整个 Attention 层（c_attn + c_proj + bias）
- mlp: 整个 MLP 层（c_fc + c_proj + bias）
- ln: 整个 LayerNorm 层（ln_1 + ln_2 + bias）
```

**实验次数**: 1 (Embedding) + 12 blocks × 3 layers × 12 sparsities = **444 次**

**预期时间**: ~2.5-3.5 小时

---

### 3. **对比工具**
**文件**: `experiments/scripts/compare_granularity.py`

**功能**: 展示粗粒度和细粒度的参数覆盖差异

**运行命令**:
```bash
python experiments/scripts/compare_granularity.py
```

---

## 🎯 使用建议

### 场景 1: 快速探索
**目标**: 快速了解整体剪枝特性

**推荐**: 使用粗粒度分析
```bash
python experiments/scripts/all_blocks_pruning_analysis.py \
    --checkpoint <your_checkpoint> \
    --device cuda
```

**优点**:
- 实验时间短（~3小时）
- 可以快速了解哪些 block 更耐剪枝

---

### 场景 2: 深入分析（你的需求）
**目标**: 了解 Attention 和 MLP 内部子层的差异

**推荐**: 使用细粒度分析 ✨
```bash
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint <your_checkpoint> \
    --device cuda
```

**优点**:
- 可以发现 QKV vs Proj 的差异
- 可以发现 FC vs Proj 的差异
- 只测试 weight，更符合实际剪枝场景

**你可以回答的问题**:
1. QKV 投影矩阵和输出投影矩阵哪个更耐剪枝？
2. MLP 的扩展层和投影层哪个更重要？
3. Attention 前的 LN 和 MLP 前的 LN 有什么差异？
4. 不同 block 的同一子层有什么差异？

---

### 场景 3: 对比分析
**目标**: 理解两种方法的区别

**推荐**: 先运行对比工具
```bash
python experiments/scripts/compare_granularity.py
```

然后分别运行两个分析脚本，对比结果。

---

## 📈 预期发现

### 细粒度分析可能发现的模式

#### 1. Attention 层内部差异
```
可能的发现:
- QKV 投影矩阵（c_attn）: 25-30% 无损剪枝
- 输出投影矩阵（c_proj）: 35-40% 无损剪枝

结论: 输出投影可能更耐剪枝
```

#### 2. MLP 层内部差异
```
可能的发现:
- 扩展层（c_fc）: 50-55% 无损剪枝
- 投影层（c_proj）: 55-60% 无损剪枝

结论: MLP 整体耐剪枝，投影层略优
```

#### 3. LayerNorm 差异
```
可能的发现:
- LN 1 (Attention 前): 45-50% 无损剪枝
- LN 2 (MLP 前): 50-55% 无损剪枝

结论: MLP 前的 LN 可能更耐剪枝
```

#### 4. 跨 Block 模式
```
可能的发现:
- 浅层 (Block 0-3): 较敏感
- 中间层 (Block 4-8): 最耐剪枝
- 深层 (Block 9-11): 中等敏感

结论: 中间层可以应用更高的剪枝比例
```

---

## 🔧 实际应用

### 基于细粒度分析设计剪枝策略

假设你的分析结果如下：

| Layer | Block 0-3 | Block 4-8 | Block 9-11 |
|-------|-----------|-----------|------------|
| attn_qkv | 20% | 30% | 25% |
| attn_proj | 30% | 40% | 35% |
| mlp_fc | 45% | 55% | 50% |
| mlp_proj | 50% | 60% | 55% |
| ln_1 | 40% | 50% | 45% |
| ln_2 | 45% | 55% | 50% |

**设计策略**:
```python
# 非均匀剪枝策略
pruning_config = {
    'block_0_3': {
        'attn_qkv': 0.20,
        'attn_proj': 0.30,
        'mlp_fc': 0.45,
        'mlp_proj': 0.50,
    },
    'block_4_8': {
        'attn_qkv': 0.30,
        'attn_proj': 0.40,
        'mlp_fc': 0.55,
        'mlp_proj': 0.60,
    },
    'block_9_11': {
        'attn_qkv': 0.25,
        'attn_proj': 0.35,
        'mlp_fc': 0.50,
        'mlp_proj': 0.55,
    },
}
```

**预期效果**:
- 整体剪枝率: ~45-50%
- 性能损失: 最小化（每层都在无损阈值内）
- 压缩率: 显著提升

---

## 📝 输出文件对比

### 粗粒度分析输出
```
results/all_blocks_pruning_analysis/
├── all_blocks_pruning_results.csv          # 完整数据
├── lossless_sparsity_summary.csv           # 摘要（3层 × 12 blocks）
├── lossless_sparsity_heatmap.png           # 热力图（12×3）
├── block_comparison.png                    # 对比图（1×3）
└── ANALYSIS_REPORT.md                      # 分析报告
```

### 细粒度分析输出
```
results/fine_grained_pruning_analysis/
├── fine_grained_pruning_results.csv        # 完整数据
├── lossless_sparsity_summary.csv           # 摘要（6层 × 12 blocks）
├── lossless_sparsity_heatmap.png           # 热力图（12×6）
├── block_comparison.png                    # 对比图（2×3）
└── ANALYSIS_REPORT.md                      # 分析报告（含层次对比）
```

**细粒度分析报告额外包含**:
- Attention 层对比（QKV vs Proj）
- MLP 层对比（FC vs Proj）
- LayerNorm 对比（LN1 vs LN2）

---

## ⚡ 快速开始

### Step 1: 运行细粒度分析
```bash
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --num_steps 100 \
    --device cuda \
    --output_dir results/fine_grained_pruning_analysis
```

### Step 2: 查看结果
```bash
cd results/fine_grained_pruning_analysis

# 查看分析报告
cat ANALYSIS_REPORT.md

# 查看热力图
open lossless_sparsity_heatmap.png

# 查看对比图
open block_comparison.png
```

### Step 3: 分析数据
```python
import pandas as pd

# 读取摘要
df = pd.read_csv('lossless_sparsity_summary.csv')

# 按层次分组统计
print(df.groupby('layer')['max_lossless_percent'].describe())

# 找出最耐剪枝的层次
print(df.nlargest(10, 'max_lossless_percent'))

# 对比 Attention 子层
attn_qkv = df[df['layer'] == 'attn_qkv']['max_lossless_percent'].mean()
attn_proj = df[df['layer'] == 'attn_proj']['max_lossless_percent'].mean()
print(f"Attention QKV 平均: {attn_qkv:.1f}%")
print(f"Attention Proj 平均: {attn_proj:.1f}%")

# 对比 MLP 子层
mlp_fc = df[df['layer'] == 'mlp_fc']['max_lossless_percent'].mean()
mlp_proj = df[df['layer'] == 'mlp_proj']['max_lossless_percent'].mean()
print(f"MLP FC 平均: {mlp_fc:.1f}%")
print(f"MLP Proj 平均: {mlp_proj:.1f}%")
```

---

## 🎓 关键概念

### 什么是"无损剪枝"？
在本实验中，无损剪枝定义为：
```
损失增量 ≤ 0.001
```

即剪枝后的损失与基线损失的差值不超过 0.001。

### 为什么只测试 weight？
1. **实际应用**: 大多数剪枝方法只剪枝 weight，保留 bias
2. **参数占比**: weight 占据绝大部分参数（>99%）
3. **稳定性**: bias 通常对模型性能影响较大，不适合剪枝

### 重要性得分如何计算？
使用一阶近似：
```python
importance_score = |gradient * weight|
```

这是 Taylor 展开的一阶项，表示移除该参数对损失的影响。

---

## 📚 相关文档

- `README_FINE_GRAINED_PRUNING.md` - 细粒度分析详细说明
- `README_ALL_BLOCKS_PRUNING.md` - 粗粒度分析详细说明
- `compare_granularity.py` - 对比工具源码

---

## 🤔 常见问题

### Q1: 应该使用哪个脚本？
**A**: 如果你想了解 Attention 和 MLP 内部的差异，使用 `fine_grained_pruning_analysis.py`。

### Q2: 实验时间太长怎么办？
**A**: 可以减少稀疏度数量或只测试部分 blocks：
```python
# 修改脚本第475行
sparsities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # 只测试6个

# 修改脚本第532行
for block_id in range(6):  # 只测试前6个blocks
```

### Q3: 为什么细粒度分析不包含 bias？
**A**:
1. bias 参数量很小（<1%）
2. 实际剪枝通常不剪 bias
3. 简化实验，聚焦于主要参数

### Q4: 如何解读热力图？
**A**:
- 颜色越绿，表示该层越耐剪枝
- 颜色越红，表示该层越敏感
- 可以直观看出不同 block 和层次的差异

### Q5: 结果如何应用到实际剪枝？
**A**:
1. 找出每层的最大无损剪枝比例
2. 设计非均匀剪枝策略
3. 对耐剪枝的层应用更高的剪枝比例
4. 对敏感的层应用较低的剪枝比例

---

## 🚀 下一步

1. **运行细粒度分析**: 获取详细的剪枝特性数据
2. **分析结果**: 找出内部差异和模式
3. **设计策略**: 基于结果设计非均匀剪枝策略
4. **实现剪枝**: 在实际模型上应用策略
5. **验证效果**: 评估压缩率和性能

---

## 📧 需要帮助？

如果遇到问题，检查：
1. 检查点是否包含 `model_state_dict`
2. GPU 显存是否足够（建议16GB+）
3. 数据集是否正确下载
4. 参数选择是否正确（运行 `compare_granularity.py` 验证）
