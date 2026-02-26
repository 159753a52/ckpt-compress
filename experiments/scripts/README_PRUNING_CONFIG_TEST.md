# 自定义剪枝配置测试脚本使用说明

## 📋 功能

`test_pruning_config.py` 允许你为每个 block 的每个参数组指定不同的剪枝比例，测试剪枝前后的损失变化。

### 核心功能
- ✅ 灵活的剪枝配置（支持 JSON 文件或命令行参数）
- ✅ 测试剪枝前后的损失变化
- ✅ 自动计算重要性得分
- ✅ 导出剪枝配置和详细统计
- ✅ 生成测试报告

---

## 🚀 快速开始

### 方法 1: 使用你的配置（Block 0）

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

### 方法 2: 使用配置文件

```bash
python experiments/scripts/test_pruning_config.py \
    --checkpoint checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt \
    --config experiments/configs/pruning_config_block0.json \
    --device cuda \
    --output_dir results/pruning_test_block0
```

---

## 📝 配置格式

### JSON 配置文件格式

#### 单个 Block 配置
```json
{
  "blocks": {
    "0": {
      "attn_qkv": 0.25,
      "attn_proj": 0.75,
      "mlp_fc": 0.45,
      "mlp_proj": 0.45,
      "ln_1": 0.25,
      "ln_2": 0.10
    }
  }
}
```

#### 多个 Blocks 配置
```json
{
  "blocks": {
    "0": {
      "attn_qkv": 0.25,
      "attn_proj": 0.75,
      "mlp_fc": 0.45,
      "mlp_proj": 0.45,
      "ln_1": 0.25,
      "ln_2": 0.10
    },
    "1": {
      "attn_qkv": 0.30,
      "attn_proj": 0.40,
      "mlp_fc": 0.50,
      "mlp_proj": 0.55,
      "ln_1": 0.30,
      "ln_2": 0.35
    }
  }
}
```

### 层次类型说明

| 层次类型 | 说明 | 参数名 | 形状 |
|---------|------|--------|------|
| `attn_qkv` | Attention QKV 投影矩阵 | `transformer.h.{block_id}.attn.c_attn.weight` | [768, 2304] |
| `attn_proj` | Attention 输出投影矩阵 | `transformer.h.{block_id}.attn.c_proj.weight` | [768, 768] |
| `mlp_fc` | MLP 第一层（扩展层） | `transformer.h.{block_id}.mlp.c_fc.weight` | [768, 3072] |
| `mlp_proj` | MLP 第二层（投影层） | `transformer.h.{block_id}.mlp.c_proj.weight` | [3072, 768] |
| `ln_1` | LayerNorm 1 (Attention 前) | `transformer.h.{block_id}.ln_1.weight` | [768] |
| `ln_2` | LayerNorm 2 (MLP 前) | `transformer.h.{block_id}.ln_2.weight` | [768] |

---

## 📊 输出文件

运行完成后，会在输出目录生成以下文件：

```
results/pruning_test_block0/
├── pruning_config.json      # 剪枝配置
├── pruning_stats.csv        # 详细统计（每层的剪枝信息）
├── summary.json             # 结果摘要
└── REPORT.md                # 测试报告
```

### 1. `pruning_config.json`
保存的剪枝配置，可以用于复现实验。

### 2. `pruning_stats.csv`
详细的剪枝统计，包含：
- `block_id`: Block 编号
- `layer_type`: 层次类型
- `param_name`: 参数名称
- `target_sparsity`: 目标稀疏度
- `actual_sparsity`: 实际稀疏度
- `total_params`: 总参数数
- `pruned_params`: 剪枝的参数数
- `kept_params`: 保留的参数数

### 3. `summary.json`
结果摘要，包含：
- `baseline_loss`: 基线损失
- `pruned_loss`: 剪枝后损失
- `loss_increase`: 损失增量
- `overall_sparsity`: 整体稀疏度
- `total_params_pruned`: 总剪枝参数数

### 4. `REPORT.md`
Markdown 格式的测试报告，包含：
- 损失对比
- 整体统计
- 剪枝配置
- 详细统计表格

---

## 🎯 使用场景

### 场景 1: 测试单个 Block 的配置
```bash
# 测试 Block 0 的特定配置
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --block 0 \
    --attn_qkv 0.25 \
    --attn_proj 0.75 \
    --mlp_fc 0.45 \
    --mlp_proj 0.45 \
    --ln_1 0.25 \
    --ln_2 0.10 \
    --device cuda
```

### 场景 2: 测试多个 Blocks 的配置
```bash
# 使用配置文件测试多个 blocks
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --config experiments/configs/pruning_config_multi_blocks.json \
    --device cuda
```

### 场景 3: 快速验证某个配置
```bash
# 减少评估批次数以加快测试
python experiments/scripts/test_pruning_config.py \
    --checkpoint <your_checkpoint> \
    --block 0 \
    --attn_qkv 0.25 \
    --attn_proj 0.75 \
    --mlp_fc 0.45 \
    --mlp_proj 0.45 \
    --ln_1 0.25 \
    --ln_2 0.10 \
    --num_eval_batches 5 \
    --device cuda
```

---

## ⏱️ 运行时间

```
梯度累积（100步）:  ~5-10 分钟
损失评估:           ~1-2 分钟
总计:               ~10-15 分钟
```

**提示**: 如果只是快速测试，可以减少 `--num_steps` 和 `--num_eval_batches`。

---

## 📈 示例输出

### 控制台输出
```
================================================================================
自定义剪枝配置测试
================================================================================
检查点: checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt
梯度累积步数: 100
评估批次数: 10
================================================================================

加载模型...
✓ 模型已加载

加载数据...
✓ 数据已加载

准备 10 个评估批次...
✓ 已准备 10 个评估批次

计算基线损失...
✓ 基线损失: 3.456789

累积梯度并计算重要性得分...
累积 100 步的梯度...
  进度: 20/100
  进度: 40/100
  进度: 60/100
  进度: 80/100
  进度: 100/100
计算重要性得分...
✓ 已计算重要性得分

================================================================================
应用剪枝配置
================================================================================

应用 Block 0 的剪枝配置:
  attn_qkv: 目标稀疏度 25.0%, 实际稀疏度 25.1%, 剪枝 443,904/1,769,472 参数
  attn_proj: 目标稀疏度 75.0%, 实际稀疏度 75.0%, 剪枝 442,368/589,824 参数
  mlp_fc: 目标稀疏度 45.0%, 实际稀疏度 45.0%, 剪枝 1,061,683/2,359,296 参数
  mlp_proj: 目标稀疏度 45.0%, 实际稀疏度 45.0%, 剪枝 1,061,683/2,359,296 参数
  ln_1: 目标稀疏度 25.0%, 实际稀疏度 25.0%, 剪枝 192/768 参数
  ln_2: 目标稀疏度 10.0%, 实际稀疏度 10.0%, 剪枝 77/768 参数

计算剪枝后的损失...

✓ 剪枝后损失: 3.457123
✓ 损失增量: +0.000334

✓ 剪枝配置已保存: results/pruning_test_block0/pruning_config.json
✓ 剪枝统计已保存: results/pruning_test_block0/pruning_stats.csv
✓ 结果摘要已保存: results/pruning_test_block0/summary.json
✓ 报告已保存: results/pruning_test_block0/REPORT.md

================================================================================
测试完成！
结果保存在: results/pruning_test_block0
================================================================================
```

### 报告示例 (REPORT.md)

```markdown
# 剪枝配置测试报告

**生成时间**: 2024-01-15 10:30:45

---

## 损失对比

- **基线损失**: 3.456789
- **剪枝后损失**: 3.457123
- **损失增量**: +0.000334
- **结论**: ✅ 无损剪枝（损失增量 ≤ 0.001）

---

## 整体统计

- **总参数数**: 7,079,424
- **剪枝参数数**: 3,009,907
- **保留参数数**: 4,069,517
- **整体稀疏度**: 42.51%

---

## 剪枝配置

```json
{
  "0": {
    "attn_qkv": 0.25,
    "attn_proj": 0.75,
    "mlp_fc": 0.45,
    "mlp_proj": 0.45,
    "ln_1": 0.25,
    "ln_2": 0.10
  }
}
```

---

## 详细统计

### Block 0

| Layer | Target Sparsity | Actual Sparsity | Pruned Params | Total Params |
|-------|-----------------|-----------------|---------------|-------------|
| attn_qkv | 25.0% | 25.1% | 443,904 | 1,769,472 |
| attn_proj | 75.0% | 75.0% | 442,368 | 589,824 |
| mlp_fc | 45.0% | 45.0% | 1,061,683 | 2,359,296 |
| mlp_proj | 45.0% | 45.0% | 1,061,683 | 2,359,296 |
| ln_1 | 25.0% | 25.0% | 192 | 768 |
| ln_2 | 10.0% | 10.0% | 77 | 768 |
```

---

## 🎓 参数说明

### 必需参数
- `--checkpoint`: 检查点路径

### 配置方式（二选一）
- `--config`: JSON 配置文件路径
- `--block` + 层次参数: 命令行配置

### 层次参数（与 --block 一起使用）
- `--attn_qkv`: Attention QKV 稀疏度 (0.0-1.0)
- `--attn_proj`: Attention Proj 稀疏度 (0.0-1.0)
- `--mlp_fc`: MLP FC 稀疏度 (0.0-1.0)
- `--mlp_proj`: MLP Proj 稀疏度 (0.0-1.0)
- `--ln_1`: LayerNorm 1 稀疏度 (0.0-1.0)
- `--ln_2`: LayerNorm 2 稀疏度 (0.0-1.0)

### 可选参数
- `--num_steps`: 梯度累积步数（默认 100）
- `--batch_size`: 批次大小（默认 4）
- `--seq_length`: 序列长度（默认 512）
- `--num_eval_batches`: 评估批次数（默认 10）
- `--device`: 设备 cuda/cpu（默认 cuda）
- `--output_dir`: 输出目录（默认 results/pruning_config_test）

---

## 💡 使用技巧

### 1. 快速测试
```bash
# 减少步数和批次以加快测试
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

### 2. 测试多个配置
```bash
# 创建多个配置文件，批量测试
for config in experiments/configs/pruning_config_*.json; do
    python experiments/scripts/test_pruning_config.py \
        --checkpoint <your_checkpoint> \
        --config $config \
        --device cuda \
        --output_dir results/$(basename $config .json)
done
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

## ⚠️ 注意事项

1. **重要性得分计算**: 每次运行都会重新计算重要性得分（约 5-10 分钟）
2. **内存需求**: 需要 16GB+ GPU 显存
3. **稀疏度范围**: 建议在 0.0-0.9 之间，过高可能导致性能严重下降
4. **评估批次**: 使用相同的评估批次确保公平对比

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
# 确认检查点包含 model_state_dict
python -c "import torch; print(torch.load('your_checkpoint.pt').keys())"
```

### 配置文件格式错误
```bash
# 验证 JSON 格式
python -m json.tool experiments/configs/pruning_config_block0.json
```

---

## 📚 相关脚本

- `fine_grained_pruning_analysis.py` - 细粒度剪枝分析（找到最优配置）
- `all_blocks_pruning_analysis.py` - 粗粒度剪枝分析
- `test_pruning_config.py` - 测试自定义配置（本脚本）

---

## 🎯 工作流建议

1. **运行细粒度分析**: 找到每层的最大无损剪枝比例
   ```bash
   python experiments/scripts/fine_grained_pruning_analysis.py \
       --checkpoint <your_checkpoint> \
       --device cuda
   ```

2. **设计剪枝配置**: 基于分析结果创建配置文件

3. **测试配置**: 使用本脚本验证配置
   ```bash
   python experiments/scripts/test_pruning_config.py \
       --checkpoint <your_checkpoint> \
       --config your_config.json \
       --device cuda
   ```

4. **迭代优化**: 根据测试结果调整配置

---

## 🎉 开始使用

测试你的配置（Block 0）:
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
    --device cuda
```

祝测试顺利！🚀
