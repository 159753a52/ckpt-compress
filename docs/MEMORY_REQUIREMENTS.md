# GPT-2 Medium 内存需求说明

## 实际内存需求分析

### 模型大小
- **参数量**: 354,823,168 (354.82M)
- **FP32 存储**: 1.32 GB
- **FP16 存储**: 0.66 GB

---

## 不同使用场景的内存需求

### 1. 推理模式（仅前向传播）✅ 最常用

```python
model = get_gpt2_medium(pretrained=True)
model.eval()
with torch.no_grad():
    outputs = model(input_ids)
```

**内存需求**:
- 模型参数: 1.32 GB
- 激活值 (batch=2, seq=512): ~0.5-1 GB
- **总计: ~2-3 GB**

**推荐配置**:
- ✅ CPU: **8GB 足够**
- ✅ GPU: **4GB 足够**

---

### 2. 训练模式（小批次）

```python
model = get_gpt2_medium(pretrained=True)
optimizer = torch.optim.Adam(model.parameters())
loss.backward()
optimizer.step()
```

**内存需求**:
- 模型参数: 1.32 GB
- 梯度: 1.32 GB
- 优化器状态 (Adam): 2.64 GB (momentum + variance)
- 激活值 (batch=4, seq=512): ~1-2 GB
- **总计: ~6-8 GB**

**推荐配置**:
- ✅ CPU: **16GB 推荐**
- ✅ GPU: **8GB 推荐**

---

### 3. AdamPrune 实验（重要性计算）

```python
# 需要计算重要性得分 + 缓存批次数据
evaluator = MemoryEfficientEvaluator(...)
importance_scores = evaluator.compute_importance_scores()
```

**内存需求**:
- 模型参数: 1.32 GB
- 梯度: 1.32 GB
- 优化器状态: 2.64 GB
- 重要性得分: 1.32 GB
- 缓存批次数据: ~0.5 GB
- 激活值: ~2 GB
- **总计: ~9-10 GB**

**推荐配置**:
- ✅ CPU: **16-24GB 推荐**
- ✅ GPU: **12-16GB 推荐**

---

## 为什么之前建议 32GB？

我之前建议 32GB 是出于以下考虑：

1. **安全余量**: 考虑到系统和其他程序占用
2. **最坏情况**: 如果使用 `deepcopy` 复制模型（会翻倍内存）
3. **大批次**: 如果使用较大的 batch_size 和 seq_length

但实际上：
- ✅ **推理**: 8GB 完全足够
- ✅ **训练**: 16GB 足够
- ✅ **AdamPrune**: 16-24GB 足够
- ⚠️ **32GB**: 仅在需要大批次或多个模型同时加载时需要

---

## 内存优化技巧

### 1. 使用 FP16 混合精度（减半内存）

```python
from torch.cuda.amp import autocast

with autocast():
    outputs = model(input_ids)
```

**效果**: 模型内存从 1.32 GB → 0.66 GB

### 2. 减小批次大小

```python
# 大批次（需要更多内存）
batch_size = 16, seq_length = 1024  # ~8 GB 激活值

# 小批次（节省内存）
batch_size = 2, seq_length = 512    # ~1 GB 激活值
```

### 3. 使用梯度累积（模拟大批次）

```python
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    loss = model(**batch).loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**效果**: 用小批次内存实现大批次效果

### 4. 使用 MemoryEfficientEvaluator（避免 deepcopy）

```python
# ❌ 不推荐: deepcopy 会翻倍内存
model_copy = copy.deepcopy(model)  # 需要额外 1.32 GB

# ✅ 推荐: 原地修改 + 恢复
evaluator = MemoryEfficientEvaluator(model, ...)
```

**效果**: 节省 1.32 GB 内存

### 5. 限制数据集大小

```python
dataloader = get_wikitext2_dataloader(
    max_samples=100,  # 仅使用 100 个样本
)
```

---

## 实际测试结果

我在你的系统上运行测试时的实际内存占用：

```bash
python scripts/test_gpt2_medium.py
```

**观察到的内存使用**:
- 模型加载: ~2 GB
- 前向传播: ~3 GB
- 状态字典保存: ~4 GB

**结论**: 8GB 内存完全可以运行基本功能！

---

## 推荐配置（更新）

### 最小配置（推理）
- CPU: 8GB
- GPU: 4GB
- 用途: 模型加载、前向传播、评估

### 推荐配置（训练）
- CPU: 16GB
- GPU: 8GB
- 用途: 模型训练、微调、小规模实验

### 理想配置（研究）
- CPU: 24-32GB
- GPU: 16GB+
- 用途: AdamPrune 实验、大批次训练、多模型对比

---

## 常见问题

### Q: 我只有 8GB 内存，能用吗？
**A**: 可以！推理模式完全没问题。训练时需要：
- 减小批次大小 (batch_size=1-2)
- 减小序列长度 (seq_length=256)
- 使用梯度累积

### Q: 我有 16GB 内存，能做什么？
**A**: 可以做大部分事情：
- ✅ 推理和评估
- ✅ 小批次训练
- ✅ AdamPrune 实验（小批次）
- ⚠️ 大批次训练需要优化

### Q: 什么时候真的需要 32GB？
**A**: 以下情况：
- 同时加载多个大模型
- 使用非常大的批次 (batch_size > 16)
- 需要缓存大量数据
- 运行多个实验进程

---

## 总结

**之前的建议过于保守！实际需求：**

| 使用场景 | CPU 内存 | GPU 显存 | 说明 |
|---------|---------|---------|------|
| 推理 | 8GB ✅ | 4GB ✅ | 完全足够 |
| 训练（小批次） | 16GB ✅ | 8GB ✅ | 推荐配置 |
| AdamPrune 实验 | 16-24GB | 12-16GB | 理想配置 |
| 大规模实验 | 32GB+ | 24GB+ | 可选 |

**关键**: 使用内存优化技巧（FP16、小批次、梯度累积、MemoryEfficientEvaluator）可以在更小的内存下运行！

---

**更新时间**: 2025-01-17
**感谢指正**: 之前的 32GB 建议确实过于保守 😊
