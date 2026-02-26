# BERT 细粒度剪枝分析脚本调试修改说明

## 问题描述

原始脚本运行后，所有稀疏度下的准确率都完全相同（0.9375），即使实际剪枝比例从 5% 到 90% 不等。这是不合理的，说明剪枝逻辑存在问题。

## 修改内容

### 1. `collect_gradients_and_compute_importance` 函数（第 161-290 行）

**添加的调试信息**：

- **记录没有梯度的参数**：
  ```python
  params_without_grad = []  # 记录没有梯度的参数
  ```
  在梯度累积时，如果某个参数在第一步没有梯度，会被记录下来。

- **警告输出**：
  ```python
  if params_without_grad:
      print(f"\n⚠️  警告: 以下 {len(params_without_grad)} 个参数在第一步没有梯度:")
  ```
  这可以帮助识别 Embedding 层或其他层是否没有梯度。

- **重要性得分统计**：
  ```python
  zero_score_params = []
  for name, score in scores.items():
      if (score == 0).all():
          zero_score_params.append(name)
  ```
  检查哪些参数的重要性得分全为 0，这是导致无法剪枝的根本原因。

### 2. `prune_and_evaluate` 函数（第 382-530 行）

**添加的调试参数**：
- 新增 `debug=False` 参数，用于控制是否输出详细调试信息

**添加的调试信息**：

1. **重要性得分分布检查**：
   ```python
   if debug:
       print(f"\n  === 调试：重要性得分分布 ===")
       for name in layer_params:
           score = layer_scores[name]
           print(f"  {name}:")
           print(f"    min={score.min():.6e}, max={score.max():.6e}, mean={score.mean():.6e}")
           print(f"    零值数量: {(score == 0).sum().item()}/{score.numel()}")
   ```
   显示每个参数的重要性得分范围，帮助识别是否所有得分都是 0。

2. **阈值计算检查**：
   ```python
   if debug:
       print(f"  阈值 (sparsity={sparsity*100:.1f}%): {threshold:.6e}")
       print(f"  得分范围: [{flat_scores.min():.6e}, {flat_scores.max():.6e}]")

       if flat_scores.min() == flat_scores.max():
           print(f"  ⚠️  警告: 所有重要性得分都相同！无法有效剪枝")
   ```
   检查阈值是否合理，以及是否所有得分都相同。

3. **剪枝操作详细跟踪**：
   ```python
   if debug:
       # 剪枝前
       before_zeros = (param.data == 0).sum().item()
       before_nonzeros = (param.data != 0).sum().item()
       before_mean = param.data.abs().mean().item()

       # 应用剪枝
       param.data.mul_(mask)

       # 剪枝后
       after_zeros = (param.data == 0).sum().item()
       after_nonzeros = (param.data != 0).sum().item()
       after_mean = param.data.abs().mean().item()

       print(f"  {name}:")
       print(f"    剪枝前: {before_zeros:,} 个零, {before_nonzeros:,} 个非零, |mean|={before_mean:.6e}")
       print(f"    剪枝后: {after_zeros:,} 个零, {after_nonzeros:,} 个非零, |mean|={after_mean:.6e}")
       print(f"    新增零: {after_zeros - before_zeros:,} ({(after_zeros - before_zeros)/param.data.numel()*100:.2f}%)")
   ```
   **这是最关键的调试信息**，可以验证：
   - 剪枝操作是否真的修改了参数
   - 修改的幅度是否符合预期

4. **评估前的状态检查**：
   ```python
   if debug:
       print(f"\n  === 调试：评估模型 ===")
       print(f"  模型模式: {'train' if model.training else 'eval'}")
       print(f"  评估批次数: {len(eval_batches)}")
   ```

5. **权重恢复验证**：
   ```python
   if debug:
       print(f"\n  === 调试：权重已恢复 ===")
       for name in layer_params:
           param = param_dict[name]
           restored_zeros = (param.data == 0).sum().item()
           print(f"  {name}: {restored_zeros:,} 个零（应该恢复到原始状态）")
   ```
   验证权重是否正确恢复到原始状态。

### 3. `evaluate_metric` 函数（第 340-409 行）

**关键修改**：
- **添加了 `model.train()` 恢复训练模式**：
  ```python
  # 恢复训练模式（重要！）
  model.train()
  ```
  这确保评估后模型回到训练模式，避免 Dropout 等层的状态影响后续操作。

### 4. 主实验循环（第 915-1005 行）

**修改**：
- **选择性启用调试模式**：
  ```python
  # Embedding 层：只在第一次稀疏度时启用详细调试
  debug_mode = (idx == 0)

  # Block 0：只在第一个层的第一次稀疏度时启用详细调试
  debug_mode = (block_id == 0 and idx == 0 and layer_type == 'attn_q')
  ```
  这样可以避免输出过多的调试信息，同时保留关键的调试数据。

## 如何使用

运行修改后的脚本：

```bash
python experiments/scripts/bert_fine_grained_pruning_analysis.py \
    --dataset sst2 \
    --num_steps 10 \
    --batch_size 16 \
    --num_eval_batches 10 \
    --device cuda
```

## 预期的调试输出

### 1. 梯度累积阶段

如果有参数没有梯度，会看到：
```
⚠️  警告: 以下 X 个参数在第一步没有梯度:
  - bert.embeddings.word_embeddings.weight
  - ...
```

如果有参数的重要性得分全为 0，会看到：
```
⚠️  警告: 以下 X 个参数的重要性得分全为 0:
  - bert.embeddings.word_embeddings.weight
  - ...
```

### 2. Embedding 层剪枝（第一次稀疏度 5%）

```
=== 调试：重要性得分分布 ===
  bert.embeddings.word_embeddings.weight:
    min=0.000000e+00, max=1.234567e-03, mean=5.678901e-05
    零值数量: 12345/30522000

  阈值 (sparsity=5.0%): 1.234567e-05
  得分范围: [0.000000e+00, 1.234567e-03]

=== 调试：剪枝操作 ===
  bert.embeddings.word_embeddings.weight:
    剪枝前: 0 个零, 30,522,000 个非零, |mean|=1.234567e-02
    剪枝后: 1,526,100 个零, 28,995,900 个非零, |mean|=1.234567e-02
    新增零: 1,526,100 (5.00%)

=== 调试：评估模型 ===
  模型模式: eval
  评估批次数: 10
  评估结果: 0.9375

=== 调试：权重已恢复 ===
  bert.embeddings.word_embeddings.weight: 0 个零（应该恢复到原始状态）
```

### 3. Block 0 Attention Query（第一次稀疏度 5%）

类似的调试输出，但针对 `bert.encoder.layer.0.attention.self.query.weight`。

## 可能的问题诊断

### 问题 1：所有重要性得分都是 0

**症状**：
```
⚠️  警告: 所有重要性得分都相同！无法有效剪枝
```

**原因**：
- 参数没有梯度（Embedding 层可能没有被激活）
- 权重全为 0（不太可能）

**解决方案**：
- 增加 `--num_steps` 到 50 或 100
- 检查模型是否正确加载了检查点

### 问题 2：剪枝操作没有生效

**症状**：
```
剪枝前: 0 个零, 1,048,576 个非零
剪枝后: 0 个零, 1,048,576 个非零
新增零: 0 (0.00%)
```

**原因**：
- Mask 全为 1（所有参数都被保留）
- 参数修改没有生效（设备不匹配）

**解决方案**：
- 检查重要性得分分布
- 检查阈值计算是否正确

### 问题 3：准确率完全相同

**症状**：
所有稀疏度下准确率都是 0.9375

**可能原因**：
1. **评估批次太少**：只有 10 个批次（160 个样本），统计噪声大
2. **模型状态缓存**：`model.eval()` 可能缓存了某些计算
3. **剪枝没有生效**：参数实际上没有被修改

**解决方案**：
- 增加 `--num_eval_batches` 到 50 或 100
- 检查调试输出中的"新增零"数量
- 验证剪枝前后的参数均值变化

## 下一步

1. **运行修改后的脚本**，查看调试输出
2. **检查 Embedding 层**是否有梯度和非零重要性得分
3. **检查 Block 0 Attention Query**的剪枝操作是否生效
4. **根据调试输出**确定问题的根本原因

## 预期结果

如果剪枝逻辑正确，应该看到：
- 随着稀疏度增加，准确率逐渐下降
- 剪枝 5-20% 时，准确率下降很小（< 0.5%）
- 剪枝 50% 以上时，准确率明显下降
- 剪枝 90% 时，准确率应该大幅下降（可能降到 50% 左右）

如果所有稀疏度下准确率都相同，说明剪枝操作没有真正影响模型的前向传播。
