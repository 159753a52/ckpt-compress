# 总结：细粒度剪枝分析工具

## ✅ 已完成的工作

### 1. 创建了细粒度剪枝分析脚本
**文件**: `experiments/scripts/fine_grained_pruning_analysis.py`

**核心改进**:
- ✅ **只测试 weight 参数**（排除所有 bias）
- ✅ **细粒度层次划分**：将 Attention 拆分为 QKV + Proj，MLP 拆分为 FC + Proj
- ✅ **6个子层**：attn_qkv, attn_proj, mlp_fc, mlp_proj, ln_1, ln_2

### 2. 创建了配套文档
- `README_FINE_GRAINED_PRUNING.md` - 详细使用说明
- `PRUNING_TOOLS_SUMMARY.md` - 工具总结和使用建议
- `compare_granularity.py` - 对比工具

---

## 🎯 你的问题已解决

### 原始问题
> "对于Attention层和MLP层其实还不够细，里面还有qkv层和输出层，有更细的层次。我希望看一下这些更细的层次不同剪枝比例的表现。而且我希望你只关注weight层次。"

### 解决方案

#### ✅ 问题1: Attention 层不够细
**解决**: 拆分为 2 个子层
```
attn_qkv:  c_attn.weight [768, 2304]  - QKV 投影矩阵
attn_proj: c_proj.weight [768, 768]   - 输出投影矩阵
```

#### ✅ 问题2: MLP 层不够细
**解决**: 拆分为 2 个子层
```
mlp_fc:   c_fc.weight [768, 3072]    - 第一层（扩展层）
mlp_proj: c_proj.weight [3072, 768]  - 第二层（投影层）
```

#### ✅ 问题3: 只关注 weight
**解决**: `select_layer_params` 函数只选择 `.weight` 参数
```python
# 示例：只选择 c_attn.weight，不包含 c_attn.bias
if f'transformer.h.{block_id}.attn.c_attn.weight' in n
```

---

## 📊 对比：粗粒度 vs 细粒度

| 维度 | 粗粒度 | 细粒度 |
|------|--------|--------|
| **Attention** | 整体（4个参数） | QKV + Proj（2个 weight） |
| **MLP** | 整体（4个参数） | FC + Proj（2个 weight） |
| **LayerNorm** | 整体（4个参数） | LN1 + LN2（2个 weight） |
| **参数类型** | weight + bias | **只 weight** |
| **层次数** | 3 | 6 |
| **实验次数** | 432 | 864 |
| **运行时间** | ~3小时 | ~4小时 |

---

## 🚀 如何使用

### 运行细粒度分析
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

### 查看对比
```bash
# 运行对比工具，了解两种方法的区别
python experiments/scripts/compare_granularity.py
```

---

## 📈 你将获得的结果

### 1. CSV 数据
- `fine_grained_pruning_results.csv` - 完整实验数据（876行）
- `lossless_sparsity_summary.csv` - 每层最大无损剪枝比例

### 2. 可视化
- `lossless_sparsity_heatmap.png` - 热力图（12 blocks × 6 layers）
- `block_comparison.png` - 6个子图对比不同层次

### 3. 分析报告
- `ANALYSIS_REPORT.md` - 包含：
  - 每层统计信息（平均、标准差、最小、最大）
  - 最耐剪枝和最敏感的 block
  - **Attention 层对比**（QKV vs Proj）
  - **MLP 层对比**（FC vs Proj）
  - **LayerNorm 对比**（LN1 vs LN2）

---

## 🔍 你可以回答的问题

### Attention 层
1. QKV 投影矩阵和输出投影矩阵哪个更耐剪枝？
2. 不同 block 的 QKV 有什么差异？
3. 输出投影在不同深度的表现如何？

### MLP 层
1. 扩展层（FC）和投影层（Proj）哪个更重要？
2. 两层的剪枝容忍度差异有多大？
3. 是否可以对某一层应用更高的剪枝比例？

### LayerNorm 层
1. Attention 前的 LN 和 MLP 前的 LN 有什么差异？
2. 哪个 LN 更耐剪枝？

### 跨 Block 分析
1. 浅层、中间层、深层的差异？
2. 哪些 block 最适合剪枝？
3. 是否存在某种模式（如中间层更耐剪枝）？

---

## 💡 实际应用示例

假设你的分析发现：
```
Attention QKV:  平均 25% 无损剪枝
Attention Proj: 平均 35% 无损剪枝
MLP FC:         平均 50% 无损剪枝
MLP Proj:       平均 55% 无损剪枝
```

**设计非均匀剪枝策略**:
```python
pruning_strategy = {
    'attn_qkv': 0.25,   # 较保守
    'attn_proj': 0.35,  # 中等
    'mlp_fc': 0.50,     # 激进
    'mlp_proj': 0.55,   # 最激进
}

# 预期整体剪枝率: ~40%
# 预期性能损失: 最小化（每层都在无损阈值内）
```

---

## 📁 文件清单

### 新创建的文件
```
experiments/scripts/
├── fine_grained_pruning_analysis.py          # 细粒度分析脚本 ⭐
├── README_FINE_GRAINED_PRUNING.md            # 详细使用说明
├── compare_granularity.py                    # 对比工具
└── PRUNING_TOOLS_SUMMARY.md                  # 工具总结
```

### 现有文件（参考）
```
experiments/scripts/
├── all_blocks_pruning_analysis.py            # 粗粒度分析脚本
└── README_ALL_BLOCKS_PRUNING.md              # 粗粒度说明
```

---

## ⚙️ 技术细节

### 参数选择逻辑
```python
def select_layer_params(layer_group, block_id, all_param_names):
    """只选择 weight 参数"""
    if layer_group == 'attn_qkv':
        # 只选择 c_attn.weight，不包含 c_attn.bias
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.attn.c_attn.weight' in n]

    elif layer_group == 'attn_proj':
        # 只选择 c_proj.weight，不包含 c_proj.bias
        return [n for n in all_param_names
                if f'transformer.h.{block_id}.attn.c_proj.weight' in n]

    # ... 其他层次类似
```

### 剪枝流程
```python
1. 计算重要性得分: importance = |gradient * weight|
2. 计算阈值: threshold = quantile(scores, sparsity)
3. 创建掩码: mask = (scores >= threshold)
4. 应用剪枝: weight *= mask
5. 评估损失: loss_after_pruning
6. 恢复权重: weight = original_weight
7. 记录结果: loss_increase = loss_after - loss_baseline
```

---

## 🎓 关键概念

### 无损剪枝
```
定义: 损失增量 ≤ 0.001
即: loss_after_pruning - loss_baseline ≤ 0.001
```

### 重要性得分
```
一阶近似: |gradient * weight|
物理意义: 移除该参数对损失的影响
```

### 稀疏度
```
定义: 被剪枝的参数比例
例如: 30% 稀疏度 = 剪枝 30% 的参数
```

---

## ✨ 主要优势

### 相比粗粒度分析
1. **更精确**: 可以发现子层之间的差异
2. **更实用**: 只测试 weight，符合实际应用
3. **更灵活**: 可以为每个子层设计不同的剪枝策略

### 相比手动分析
1. **自动化**: 一次运行测试所有层次
2. **系统化**: 统一的实验设置和评估标准
3. **可视化**: 自动生成热力图和对比图

---

## 📝 下一步建议

1. **运行实验**: 使用你的检查点运行细粒度分析
2. **分析结果**: 查看报告，找出内部差异
3. **设计策略**: 基于结果设计非均匀剪枝策略
4. **实现剪枝**: 在实际模型上应用策略
5. **验证效果**: 评估压缩率和性能保持

---

## 🔗 相关资源

- **使用说明**: `README_FINE_GRAINED_PRUNING.md`
- **工具总结**: `PRUNING_TOOLS_SUMMARY.md`
- **对比工具**: `compare_granularity.py`
- **粗粒度分析**: `all_blocks_pruning_analysis.py`

---

## 🎉 总结

你现在拥有一个完整的细粒度剪枝分析工具，可以：

✅ 分析 Attention 层内部（QKV vs Proj）
✅ 分析 MLP 层内部（FC vs Proj）
✅ 分析 LayerNorm 层内部（LN1 vs LN2）
✅ 只关注 weight 参数
✅ 自动生成详细报告和可视化
✅ 为每个子层设计最优剪枝策略

**开始使用**:
```bash
python experiments/scripts/fine_grained_pruning_analysis.py \
    --checkpoint <your_checkpoint> \
    --device cuda
```

祝实验顺利！🚀
