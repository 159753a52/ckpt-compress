# GPT-2 Medium 模型使用指南

## 概述

本项目已成功集成 GPT-2 Medium 预训练模型，可用于检查点压缩实验和 AdamPrune 方法研究。

## 模型信息

- **模型名称**: GPT-2 Medium
- **参数量**: 354.82M (354,823,168 参数)
- **架构配置**:
  - 层数 (n_layer): 24
  - 隐藏维度 (n_embd): 1024
  - 注意力头数 (n_head): 16
  - 词汇表大小 (vocab_size): 50257
  - 最大序列长度 (n_positions): 1024
- **模型大小**: ~1.5GB
- **缓存位置**: `./data/models/models--gpt2-medium/`

## 快速开始

### 1. 下载模型

使用提供的下载脚本：

```bash
# 仅下载 GPT-2 Medium
python scripts/download_models.py --model gpt2-medium

# 指定缓存目录
python scripts/download_models.py --model gpt2-medium --cache_dir ./data/models

# 下载所有模型（包括 GPT-2 Small 和 ResNet-50）
python scripts/download_models.py --all
```

### 2. 加载模型

在 Python 代码中使用：

```python
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

# 加载预训练模型
model = get_gpt2_medium(pretrained=True)

# 或者创建随机初始化的模型（用于测试）
model = get_gpt2_medium(pretrained=False)
```

### 3. 基本使用

```python
import torch
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

# 加载模型
model = get_gpt2_medium(pretrained=True)
model.eval()

# 创建输入
batch_size = 2
seq_length = 128
input_ids = torch.randint(0, 50257, (batch_size, seq_length))

# 前向传播
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits  # 形状: (batch_size, seq_length, vocab_size)

print(f"输出形状: {logits.shape}")  # torch.Size([2, 128, 50257])
```

## 高级用法

### 1. 文本生成

```python
from transformers import GPT2Tokenizer
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

# 加载模型和 tokenizer
model = get_gpt2_medium(pretrained=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

# 输入文本
prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 生成文本
model.eval()
with torch.no_grad():
    outputs = model.model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )

# 解码
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### 2. 计算损失（语言模型训练）

```python
import torch
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

model = get_gpt2_medium(pretrained=True)
model.train()

# 准备数据（input_ids 和 labels 相同，模型内部会自动移位）
input_ids = torch.randint(0, 50257, (2, 128))
labels = input_ids.clone()

# 前向传播（自动计算损失）
outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(f"Loss: {loss.item()}")
```

### 3. 保存和加载状态字典

```python
import torch
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

# 加载模型
model = get_gpt2_medium(pretrained=True)

# 保存状态字典
state_dict = model.state_dict()
torch.save(state_dict, './checkpoints/gpt2_medium.pt')

# 加载状态字典
loaded_state_dict = torch.load('./checkpoints/gpt2_medium.pt')
model.load_state_dict(loaded_state_dict)
```

### 4. 与数据加载器配合使用

```python
from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader
from src.ckpt_compress.models.gpt2 import get_gpt2_medium
import torch

# 加载模型
model = get_gpt2_medium(pretrained=True)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 加载数据
dataloader = get_wikitext2_dataloader(
    split='validation',
    batch_size=4,
    seq_length=512,
    max_samples=100,
)

# 评估困惑度
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        total_loss += loss.item() * input_ids.size(0)
        total_tokens += input_ids.size(0)

avg_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(avg_loss))
print(f"Perplexity: {perplexity.item():.2f}")
```

## 用于 AdamPrune 实验

### 1. 计算重要性得分

```python
from src.ckpt_compress.methods.adam_prune.memory_efficient import (
    MemoryEfficientEvaluator
)
from src.ckpt_compress.models.gpt2 import get_gpt2_medium
from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader
import torch

# 加载模型和数据
model = get_gpt2_medium(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

dataloader = get_wikitext2_dataloader(
    split='train',
    batch_size=4,
    seq_length=512,
    max_samples=100,
)

# 创建优化器
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# 创建评估器
evaluator = MemoryEfficientEvaluator(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    num_batches=10,
)

# 缓存批次数据
evaluator.cache_batches()

# 计算重要性得分
importance_scores = evaluator.compute_importance_scores()

print(f"计算了 {len(importance_scores)} 个参数的重要性得分")
```

### 2. 自适应剪枝

```python
from src.ckpt_compress.methods.adam_prune.adaptive_pruning import (
    adaptive_prune_model
)
from src.ckpt_compress.models.gpt2 import get_gpt2_medium
import torch

# 加载模型
model = get_gpt2_medium(pretrained=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 准备数据和优化器
# ... (同上)

# 自适应剪枝
pruned_model, results = adaptive_prune_model(
    model=model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    target_sparsity=0.5,  # 目标稀疏度 50%
    max_loss_increase=0.1,  # 最大损失增加 10%
)

print(f"剪枝后稀疏度: {results['final_sparsity']:.2%}")
print(f"损失增加: {results['loss_increase']:.4f}")
```

## 实验脚本

项目提供了多个实验脚本，可直接使用 GPT-2 Medium：

### 1. 公式对比实验

```bash
python experiments/scripts/run_formula_comparison.py \
  --mode comparison \
  --max_samples 100 \
  --batch_size 4 \
  --num_batches 10
```

### 2. 幂律校正实验

```bash
python experiments/scripts/run_formula_comparison.py \
  --mode calibration \
  --max_samples 100 \
  --batch_size 4
```

### 3. 自适应剪枝实验

```bash
python experiments/scripts/run_adaptive_pruning.py \
  --max_samples 100 \
  --batch_size 4 \
  --target_sparsity 0.5
```

## 内存管理建议

GPT-2 Medium 是一个大模型（~355M 参数），需要注意内存管理：

### 推荐配置

- **CPU 内存**: 至少 32GB
- **GPU 显存**: 至少 8GB（推荐 16GB+）
- **批次大小**: 2-8（根据显存调整）
- **序列长度**: 512（可根据需要调整）

### 内存优化技巧

1. **使用 MemoryEfficientEvaluator**（避免 deepcopy）:
   ```python
   from src.ckpt_compress.methods.adam_prune.memory_efficient import (
       MemoryEfficientEvaluator
   )
   ```

2. **限制数据集大小**:
   ```python
   dataloader = get_wikitext2_dataloader(
       split='train',
       max_samples=100,  # 限制样本数
   )
   ```

3. **使用梯度累积**:
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = model(**batch).loss / accumulation_steps
       loss.backward()

       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **使用混合精度训练**:
   ```python
   from torch.cuda.amp import autocast, GradScaler

   scaler = GradScaler()

   with autocast():
       outputs = model(**batch)
       loss = outputs.loss

   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

## 测试

运行测试脚本验证模型功能：

```bash
# 运行完整测试
python scripts/test_gpt2_medium.py

# 运行单元测试
pytest tests/unit/models/test_gpt2.py -v
```

## 常见问题

### Q1: 模型下载失败怎么办？

**A**: 如果遇到网络问题，可以：
1. 使用镜像源（如 HuggingFace 镜像）
2. 手动下载模型文件到 `./data/models/models--gpt2-medium/`
3. 使用代理

### Q2: 内存不足（OOM）怎么办？

**A**: 尝试以下方法：
1. 减小批次大小（`batch_size=2` 或 `1`）
2. 减小序列长度（`seq_length=256` 或 `128`）
3. 使用 `MemoryEfficientEvaluator` 而不是 `deepcopy`
4. 限制数据集大小（`max_samples=50`）
5. 使用 CPU 而不是 GPU（速度较慢但内存更大）

### Q3: 如何使用本地已下载的模型？

**A**: 模型会自动缓存到 `./data/models/`，下次加载时会直接使用缓存：
```python
# 第一次会下载
model = get_gpt2_medium(pretrained=True)

# 第二次会使用缓存，不会重新下载
model = get_gpt2_medium(pretrained=True)
```

### Q4: 如何查看模型结构？

**A**: 使用以下代码：
```python
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

model = get_gpt2_medium(pretrained=True)

# 打印模型结构
print(model)

# 打印参数名称和形状
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

## 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目快速参考
- [ARCHITECTURE.md](./ARCHITECTURE.md) - 架构设计
- [EXPERIMENTS.md](./EXPERIMENTS.md) - 实验指南
- [DATA_PREPARATION.md](./DATA_PREPARATION.md) - 数据准备

## 参考资料

- [GPT-2 论文](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [HuggingFace GPT-2 文档](https://huggingface.co/gpt2-medium)
- [Transformers 库文档](https://huggingface.co/docs/transformers/)
