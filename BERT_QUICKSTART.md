# 快速开始：使用 BERT-Large 模型

本指南帮助你快速开始使用项目中新添加的 BERT-Large 模型。

## 1️⃣ 下载模型

```bash
# 下载 BERT-Large 预训练模型（约 1.3GB）
python scripts/download_models.py --model bert-large

# 或者下载 BERT-Base 和 BERT-Large
python scripts/download_models.py --model bert-base,bert-large
```

## 2️⃣ 在代码中使用

```python
from src.ckpt_compress.models.bert import get_bert_large
import torch

# 加载预训练模型
model = get_bert_large(pretrained=True, cache_dir="./data/models")
model.eval()

# 创建输入
batch_size = 2
seq_length = 128
input_ids = torch.randint(0, 30522, (batch_size, seq_length))
attention_mask = torch.ones(batch_size, seq_length)

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    print(f"输出形状: {logits.shape}")  # (2, 128, 30522)
```

## 3️⃣ 运行示例

```bash
# 运行完整测试
python examples/test_bert_large.py

# 运行快速开始示例
python examples/bert_quickstart.py --all
```

## 📚 更多信息

- **详细使用指南**: [docs/BERT_USAGE.md](docs/BERT_USAGE.md)
- **集成总结**: [docs/BERT_INTEGRATION_SUMMARY.md](docs/BERT_INTEGRATION_SUMMARY.md)
- **项目文档**: [CLAUDE.md](CLAUDE.md)

## 🔑 关键特性

- ✅ 支持 BERT-Base (~110M) 和 BERT-Large (~340M)
- ✅ 自动从 HuggingFace 下载预训练权重
- ✅ 支持离线模式（本地文件）
- ✅ 与现有压缩方法无缝集成
- ✅ 完整的文档和示例

## 💡 常见用法

### 查看模型信息
```python
model = get_bert_large(pretrained=True)
info = model.get_model_info()
print(info)
```

### 保存和加载状态字典
```python
# 保存
state_dict = model.state_dict()
torch.save(state_dict, 'bert_large.pt')

# 加载
loaded_state_dict = torch.load('bert_large.pt')
model.load_state_dict(loaded_state_dict)
```

### 用于检查点压缩
```python
from src.ckpt_compress.methods.adam_prune import AdamPruneCompressor

compressor = AdamPruneCompressor(sparsity=0.5)
compressed = compressor.compress(model.state_dict())
restored = compressor.decompress(compressed)
```

## ⚠️ 注意事项

- BERT-Large 需要约 1.5GB GPU 内存（推理）
- 首次下载需要约 1.3GB 存储空间
- 建议使用 GPU 进行推理和训练

## 🆘 遇到问题？

查看 [docs/BERT_USAGE.md](docs/BERT_USAGE.md) 中的"常见问题"章节。
