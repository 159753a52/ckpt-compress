# BERT 模型使用指南

本文档介绍如何在 `ckpt-compress` 项目中使用 BERT-Base 和 BERT-Large 模型。

## 模型概述

### BERT-Base
- **参数量**: ~110M
- **层数**: 12
- **隐藏维度**: 768
- **注意力头数**: 12
- **中间层维度**: 3072
- **最大序列长度**: 512

### BERT-Large
- **参数量**: ~340M
- **层数**: 24
- **隐藏维度**: 1024
- **注意力头数**: 16
- **中间层维度**: 4096
- **最大序列长度**: 512

## 快速开始

### 1. 下载预训练模型

使用下载脚本下载 BERT-Large 模型：

```bash
# 仅下载 BERT-Large
python scripts/download_models.py --model bert-large

# 下载 BERT-Base 和 BERT-Large
python scripts/download_models.py --model bert-base,bert-large

# 下载所有模型（包括 BERT）
python scripts/download_models.py --all

# 指定缓存目录
python scripts/download_models.py --model bert-large --cache_dir ./my_models
```

**注意**: BERT-Large 模型约 1.3GB，首次下载可能需要一些时间。

### 2. 在代码中使用

#### 基本使用

```python
from src.ckpt_compress.models.bert import get_bert_large
import torch

# 加载预训练模型
model = get_bert_large(pretrained=True, cache_dir="./data/models")

# 设置为评估模式
model.eval()

# 创建输入
batch_size = 4
seq_length = 128
input_ids = torch.randint(0, 30522, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # 形状: (batch_size, seq_length, vocab_size)
```

#### 使用本地模型（离线模式）

如果已经下载了模型，可以使用 `local_files_only=True` 避免网络请求：

```python
model = get_bert_large(
    pretrained=True,
    local_files_only=True,
    cache_dir="./data/models"
)
```

#### 随机初始化（不使用预训练权重）

```python
# 用于测试或从头训练
model = get_bert_large(pretrained=False)
```

#### 查看模型信息

```python
model = get_bert_large(pretrained=True)

# 获取模型配置
info = model.get_model_info()
print(info)
# 输出:
# {
#     'model_size': 'large',
#     'vocab_size': 30522,
#     'hidden_size': 1024,
#     'num_hidden_layers': 24,
#     'num_attention_heads': 16,
#     'intermediate_size': 4096,
#     'max_position_embeddings': 512
# }

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
```

### 3. 运行测试脚本

我们提供了一个测试脚本来验证 BERT-Large 的安装和使用：

```bash
# 完整测试（包括下载）
python examples/test_bert_large.py

# 跳过下载测试（假设已下载）
python examples/test_bert_large.py --skip-download

# 运行 BERT-Base 和 BERT-Large 对比
python examples/test_bert_large.py --comparison
```

## 高级用法

### 1. 使用 Masked Language Modeling (MLM)

```python
from src.ckpt_compress.models.bert import get_bert_large
import torch

model = get_bert_large(pretrained=True)
model.eval()

# 创建输入（包含 [MASK] token，id=103）
input_ids = torch.tensor([[101, 2023, 2003, 103, 3231, 102]])  # [CLS] this is [MASK] text [SEP]
attention_mask = torch.ones_like(input_ids)

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = outputs.logits

# 获取 [MASK] 位置的预测
mask_position = 3
predicted_token_id = predictions[0, mask_position].argmax(dim=-1)
print(f"预测的 token ID: {predicted_token_id}")
```

### 2. 计算 MLM 损失

```python
# 创建标签（-100 表示不计算损失的位置）
labels = input_ids.clone()
labels[labels != 103] = -100  # 只计算 [MASK] 位置的损失

# 前向传播并计算损失
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels
)
loss = outputs.loss
print(f"MLM Loss: {loss.item()}")
```

### 3. 提取隐藏状态

```python
# 需要直接访问底层模型
bert_model = model.model.bert

outputs = bert_model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True
)

# 获取所有层的隐藏状态
hidden_states = outputs.hidden_states  # Tuple of (batch_size, seq_len, hidden_size)
print(f"层数: {len(hidden_states)}")
print(f"最后一层形状: {hidden_states[-1].shape}")
```

### 4. 用于检查点压缩实验

```python
from src.ckpt_compress.methods.adam_prune import AdamPruneCompressor

# 加载模型
model = get_bert_large(pretrained=True)

# 获取状态字典
state_dict = model.state_dict()

# 使用 AdamPrune 压缩
compressor = AdamPruneCompressor(
    sparsity=0.5,  # 50% 稀疏度
    quantization_bits=8
)

# 压缩
compressed_data = compressor.compress(state_dict)
print(f"原始大小: {len(str(state_dict).encode())} bytes")
print(f"压缩后大小: {len(compressed_data)} bytes")

# 解压
restored_state_dict = compressor.decompress(compressed_data)

# 加载恢复的权重
model.load_state_dict(restored_state_dict)
```

## 内存管理

BERT-Large 是一个大模型（~340M 参数），需要注意内存使用：

### GPU 内存需求

- **推理**: ~1.5GB GPU 内存（FP32）
- **训练**: ~6-8GB GPU 内存（取决于批次大小）
- **使用混合精度 (FP16)**: 可减少约 50% 内存

### 内存优化技巧

```python
# 1. 使用半精度
model = get_bert_large(pretrained=True)
model = model.half()  # 转换为 FP16

# 2. 使用梯度检查点（训练时）
model.model.gradient_checkpointing_enable()

# 3. 减小批次大小
batch_size = 2  # 而不是 8 或 16

# 4. 使用 CPU 卸载（如果 GPU 内存不足）
model = model.cpu()
```

## 与其他模型对比

| 模型 | 参数量 | 层数 | 隐藏维度 | 适用场景 |
|------|--------|------|----------|----------|
| BERT-Base | ~110M | 12 | 768 | 一般 NLP 任务 |
| BERT-Large | ~340M | 24 | 1024 | 需要更高精度的任务 |
| GPT-2 Small | ~117M | 12 | 768 | 文本生成 |
| GPT-2 Medium | ~345M | 24 | 1024 | 高质量文本生成 |

## 常见问题

### Q: 下载速度慢怎么办？

A: 可以使用镜像源：

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model = get_bert_large(pretrained=True)
```

### Q: 如何查看下载进度？

A: transformers 库会自动显示下载进度条。

### Q: 模型保存在哪里？

A: 默认保存在 `./data/models/` 目录。可以通过 `cache_dir` 参数指定其他位置。

### Q: 可以使用中文 BERT 吗？

A: 当前实现使用的是英文 BERT (`bert-large-uncased`)。如需中文 BERT，可以修改 `bert.py` 中的模型名称为 `bert-base-chinese`。

### Q: 如何在多 GPU 上使用？

A: 使用 PyTorch 的 DataParallel 或 DistributedDataParallel：

```python
model = get_bert_large(pretrained=True)
model = torch.nn.DataParallel(model)
model = model.cuda()
```

## 相关文档

- [ARCHITECTURE.md](../docs/ARCHITECTURE.md) - 项目架构说明
- [EXPERIMENTS.md](../docs/EXPERIMENTS.md) - 实验指南
- [DATA_PREPARATION.md](../docs/DATA_PREPARATION.md) - 数据准备指南

## 参考资料

- [BERT 论文](https://arxiv.org/abs/1810.04805)
- [HuggingFace BERT 文档](https://huggingface.co/docs/transformers/model_doc/bert)
- [BERT GitHub](https://github.com/google-research/bert)
