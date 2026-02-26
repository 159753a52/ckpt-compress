# ✅ 自定义剪枝配置测试工具 - 完成

## 🎉 已创建的工具

我已经为你创建了一个完整的自定义剪枝配置测试工具，可以测试你指定的剪枝配置！

---

## 📁 创建的文件

### 核心脚本
1. **`test_pruning_config.py`** ⭐ - 主测试脚本
2. **`test_block0_config.sh`** - 快速启动脚本（你的配置）

### 配置文件
3. **`experiments/configs/pruning_config_block0.json`** - 你的 Block 0 配置
4. **`experiments/configs/pruning_config_multi_blocks.json`** - 多 Block 配置示例

### 文档
5. **`README_PRUNING_CONFIG_TEST.md`** - 详细使用说明
6. **`PRUNING_CONFIG_TEST_REFERENCE.txt`** - 快速参考卡片

---

## 🎯 你的配置 (Block 0)

```json
{
  "blocks": {
    "0": {
      "attn_qkv": 0.25,    // Attention QKV 投影矩阵: 25%
      "attn_proj": 0.75,   // Attention 输出投影矩阵: 75% ⭐ 非常激进
      "mlp_fc": 0.45,      // MLP 第一层（扩展层）: 45%
      "mlp_proj": 0.45,    // MLP 第二层（投影层）: 45%
      "ln_1": 0.25,        // LayerNorm 1 (Attention 前): 25%
      "ln_2": 0.10         // LayerNorm 2 (MLP 前): 10%
    }
  }
}
```

**配置特点**:
- ✅ `attn_proj` (75%) 非常激进 - 测试输出投影的鲁棒性
- ✅ `mlp_fc/mlp_proj` (45%) 中等剪枝 - 平衡压缩和性能
- ✅ `attn_qkv` (25%) 较保守 - QKV 通常更敏感
- ✅ `ln_2` (10%) 最保守 - MLP 前的 LN 保留更多参数

---

## 🚀 如何使用

### 方法 1: 快速启动（推荐）⭐

```bash
./experiments/scripts/test_block0_config.sh
```

这个脚本会：
1. 检查检查点是否存在
2. 检测 CUDA 是否可用
3. 显示配置信息
4. 运行测试
5. 显示关键结果

### 方法 2: 使用命令行参数

```bash
python experiments/scripts/test_pruning_config.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --block 0 \
    --attn_qkv 0.25 \
    --attn_proj 0.75 \
    --mlp_fc 0.45 \
    --mlp_proj 0.45 \
    --ln_1 0.25 \
    --ln_2 0.10 \
    --device cuda \
    --output_dir results/pruning_test_block0
```

### 方法 3: 使用配置文件

```bash
python experiments/scripts/test_pruning_config.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --config experiments/configs/pruning_config_block0.json \
    --device cuda \
    --output_dir results/pruning_test_block0
```

---

## ⏱️ 运行时间

```
梯度累积（100步）:  ~5-10 分钟
损失评估:           ~1-2 分钟
总计:               ~10-15 分钟
```

**快速测试**（减少时间到 ~5 分钟）:
```bash
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --block 0 \
    --attn_qkv 0.25 \
    --attn_proj 0.75 \
    --mlp_fc 0.45 \
    --mlp_proj 0.45 \
    --ln_1 0.25 \
    --ln_2 0.10 \
    --num_steps 50 \
    --num_eval_batches 5 \
    --device cuda
```

---

## 📊 输出文件

运行完成后，会在输出目录生成：

```
results/pruning_test_block0/
├── pruning_config.json      # 剪枝配置（可复现）
├── pruning_stats.csv        # 详细统计（每层的剪枝信息）
├── summary.json             # 结果摘要（JSON 格式）
└── REPORT.md                # 测试报告（Markdown 格式）
```

### 关键输出信息

**summary.json**:
```json
{
  "baseline_loss": 3.456789,
  "pruned_loss": 3.457123,
  "loss_increase": 0.000334,
  "overall_sparsity": 0.4251,
  "total_params": 7079424,
  "total_params_pruned": 3009907
}
```

**REPORT.md** 包含:
- ✅ 损失对比（基线 vs 剪枝后）
- ✅ 整体统计（总参数、剪枝参数、整体稀疏度）
- ✅ 剪枝配置（JSON 格式）
- ✅ 详细统计表格（每层的目标/实际稀疏度）

---

## 📈 预期结果

基于你的配置，预期结果：

```
基线损失:     ~3.45-3.50
剪枝后损失:   ~3.45-3.51
损失增量:     ~0.0003-0.001
整体稀疏度:   ~42-45%
结论:         ✅ 可能无损剪枝（损失增量 ≤ 0.001）
```

**关键观察点**:
1. **attn_proj (75%)** 是否能保持无损？这是一个很激进的剪枝比例
2. **整体稀疏度** 约 42-45%，是一个不错的压缩率
3. **损失增量** 如果 ≤ 0.001，说明配置成功

---

## 💡 使用技巧

### 1. 测试不同的配置

```bash
# 测试更激进的配置
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --block 0 \
    --attn_qkv 0.30 \
    --attn_proj 0.80 \
    --mlp_fc 0.50 \
    --mlp_proj 0.60 \
    --device cuda

# 测试更保守的配置
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --block 0 \
    --attn_qkv 0.20 \
    --attn_proj 0.30 \
    --mlp_fc 0.40 \
    --mlp_proj 0.45 \
    --device cuda
```

### 2. 测试多个 Blocks

创建配置文件 `my_config.json`:
```json
{
  "blocks": {
    "0": {"attn_qkv": 0.25, "attn_proj": 0.75, ...},
    "1": {"attn_qkv": 0.30, "attn_proj": 0.40, ...},
    "2": {"attn_qkv": 0.35, "attn_proj": 0.45, ...}
  }
}
```

然后运行:
```bash
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --config my_config.json \
    --device cuda
```

### 3. 只测试部分层次

```bash
# 只测试 MLP 层
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --block 0 \
    --mlp_fc 0.50 \
    --mlp_proj 0.60 \
    --device cuda
```

---

## 🎓 工作流建议

### 完整工作流

1. **运行细粒度分析**（可选，找到最优配置）
   ```bash
   python experiments/scripts/fine_grained_pruning_analysis.py \
       --checkpoint <your_checkpoint> \
       --device cuda
   ```

2. **设计剪枝配置**
   - 基于分析结果
   - 或基于经验/直觉

3. **测试配置**
   ```bash
   ./experiments/scripts/test_block0_config.sh
   ```

4. **查看结果**
   ```bash
   cat results/pruning_test_block0/REPORT.md
   ```

5. **迭代优化**
   - 如果损失增量太大，降低剪枝比例
   - 如果损失增量很小，可以尝试更激进的配置

---

## 📊 结果解读

### 损失增量判断

| 损失增量 | 结论 | 建议 |
|---------|------|------|
| ≤ 0.001 | ✅ 无损剪枝 | 配置可接受，可以尝试更激进 |
| 0.001-0.01 | ⚠️ 轻微损失 | 可能可接受，取决于应用场景 |
| > 0.01 | ❌ 明显损失 | 需要降低剪枝比例 |

### 整体稀疏度

| 稀疏度 | 策略 | 说明 |
|--------|------|------|
| < 30% | 保守 | 安全但压缩率低 |
| 30-50% | 平衡 | 推荐范围 |
| > 50% | 激进 | 高压缩但风险大 |

---

## 🔍 故障排除

### CUDA Out of Memory
```bash
# 减小批次大小
--batch_size 2
--num_eval_batches 5
```

### 检查点加载失败
```bash
# 验证检查点
python -c "import torch; print(torch.load('your_checkpoint.pt').keys())"
```

### 配置文件格式错误
```bash
# 验证 JSON 格式
python -m json.tool experiments/configs/pruning_config_block0.json
```

---

## 📚 查看文档

```bash
# 快速参考
cat experiments/scripts/PRUNING_CONFIG_TEST_REFERENCE.txt

# 详细说明
cat experiments/scripts/README_PRUNING_CONFIG_TEST.md

# 你的配置
cat experiments/configs/pruning_config_block0.json
```

---

## 🎉 开始测试

### 立即测试你的配置

```bash
# 方法 1: 快速启动
./experiments/scripts/test_block0_config.sh

# 方法 2: 完整命令
python experiments/scripts/test_pruning_config.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --block 0 \
    --attn_qkv 0.25 \
    --attn_proj 0.75 \
    --mlp_fc 0.45 \
    --mlp_proj 0.45 \
    --ln_1 0.25 \
    --ln_2 0.10 \
    --device cuda
```

---

## 💾 内存需求

- **GPU 显存**: 16GB+
- **系统内存**: 32GB+

---

## ✨ 主要特性

1. **灵活配置**: 支持 JSON 文件或命令行参数
2. **自动计算**: 自动计算重要性得分
3. **详细统计**: 每层的目标/实际稀疏度
4. **可复现**: 保存配置文件，可以复现实验
5. **多格式输出**: JSON、CSV、Markdown

---

## 🎯 总结

你现在拥有一个完整的剪枝配置测试工具，可以：

✅ 测试你指定的 Block 0 配置
✅ 测试任意 block 的任意配置
✅ 测试多个 blocks 的联合配置
✅ 导出详细的统计和报告
✅ 快速迭代优化配置

**你的配置特点**:
- attn_proj (75%) 非常激进 - 这是一个有趣的测试
- 整体稀疏度约 42-45% - 不错的压缩率
- 预期可能无损或轻微损失

**开始测试**:
```bash
./experiments/scripts/test_block0_config.sh
```

祝测试顺利！🚀
