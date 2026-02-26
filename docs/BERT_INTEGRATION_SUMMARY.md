# BERT-Large 模型集成总结

本文档总结了为 `ckpt-compress` 项目添加 BERT-Large 模型支持所做的所有更改。

## 添加的文件

### 1. 核心模型文件
- **`src/ckpt_compress/models/bert.py`**
  - 实现了 `BERTForExperiment` 类，封装 HuggingFace 的 BERT 模型
  - 支持 BERT-Base (~110M 参数) 和 BERT-Large (~340M 参数)
  - 提供工厂函数：`get_bert_base()` 和 `get_bert_large()`
  - 支持预训练权重加载和随机初始化
  - 支持本地文件模式（离线使用）

### 2. 文档
- **`docs/BERT_USAGE.md`**
  - 完整的 BERT 模型使用指南
  - 包含快速开始、高级用法、内存管理等章节
  - 提供多个代码示例
  - 包含常见问题解答

### 3. 示例脚本
- **`examples/test_bert_large.py`**
  - 测试 BERT-Large 模型的下载和加载
  - 验证模型配置和前向传播
  - 对比 BERT-Base 和 BERT-Large

- **`examples/bert_quickstart.py`**
  - 4 个快速开始示例
  - 涵盖基本使用、模型信息、模型对比、状态字典操作

## 修改的文件

### 1. 模型模块初始化
- **`src/ckpt_compress/models/__init__.py`**
  - 添加了 BERT 模型的导出
  - 更新了 `__all__` 列表

### 2. 下载脚本
- **`scripts/download_models.py`**
  - 添加了 `download_bert_base()` 函数
  - 添加了 `download_bert_large()` 函数
  - 更新了 `download_all_models()` 中的可用模型列表
  - 更新了命令行帮助信息

### 3. 项目文档
- **`CLAUDE.md`**
  - 更新了模型下载命令示例
  - 添加了 BERT 模型到目录结构说明
  - 添加了 BERT_USAGE.md 的引用

## 功能特性

### BERT 模型封装
```python
from src.ckpt_compress.models.bert import get_bert_large

# 加载预训练模型
model = get_bert_large(pretrained=True, cache_dir="./data/models")

# 本地模式（离线使用）
model = get_bert_large(pretrained=True, local_files_only=True)

# 随机初始化
model = get_bert_large(pretrained=False)
```

### 模型下载
```bash
# 下载 BERT-Large
python scripts/download_models.py --model bert-large

# 下载 BERT-Base 和 BERT-Large
python scripts/download_models.py --model bert-base,bert-large

# 下载所有模型（包括 BERT）
python scripts/download_models.py --all
```

### 模型信息
- **BERT-Base**: ~110M 参数, 12 层, 768 隐藏维度
- **BERT-Large**: ~340M 参数, 24 层, 1024 隐藏维度

## 使用示例

### 基本使用
```python
import torch
from src.ckpt_compress.models.bert import get_bert_large

# 加载模型
model = get_bert_large(pretrained=True)
model.eval()

# 创建输入
input_ids = torch.randint(0, 30522, (2, 128))
attention_mask = torch.ones(2, 128)

# 前向传播
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
```

### 查看模型信息
```python
model = get_bert_large(pretrained=True)

# 获取配置
info = model.get_model_info()
print(info)

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

### 用于检查点压缩
```python
from src.ckpt_compress.models.bert import get_bert_large
from src.ckpt_compress.methods.adam_prune import AdamPruneCompressor

# 加载模型
model = get_bert_large(pretrained=True)
state_dict = model.state_dict()

# 压缩
compressor = AdamPruneCompressor(sparsity=0.5, quantization_bits=8)
compressed_data = compressor.compress(state_dict)

# 解压
restored_state_dict = compressor.decompress(compressed_data)
model.load_state_dict(restored_state_dict)
```

## 测试

### 运行测试脚本
```bash
# 完整测试（包括下载）
python examples/test_bert_large.py

# 跳过下载测试
python examples/test_bert_large.py --skip-download

# 运行对比测试
python examples/test_bert_large.py --comparison
```

### 运行快速开始示例
```bash
# 运行所有示例
python examples/bert_quickstart.py --all

# 运行指定示例
python examples/bert_quickstart.py --example 1
python examples/bert_quickstart.py --example 2
python examples/bert_quickstart.py --example 3
python examples/bert_quickstart.py --example 4
```

## 内存要求

### BERT-Large
- **推理**: ~1.5GB GPU 内存（FP32）
- **训练**: ~6-8GB GPU 内存（取决于批次大小）
- **模型文件**: ~1.3GB

### 内存优化
```python
# 使用半精度
model = get_bert_large(pretrained=True)
model = model.half()

# 使用梯度检查点（训练时）
model.model.gradient_checkpointing_enable()

# 减小批次大小
batch_size = 2
```

## 依赖项

BERT 模型需要以下依赖：
- `transformers` (HuggingFace)
- `torch`

这些依赖已经在项目的 `setup.py` 或 `requirements.txt` 中。

## 与现有模型的集成

BERT 模型与项目中现有的模型（ResNet、GPT-2）保持一致的接口：

| 模型 | 参数量 | 层数 | 隐藏维度 | 用途 |
|------|--------|------|----------|------|
| ResNet-18 | ~11M | 18 | - | 图像分类 |
| ResNet-50 | ~25M | 50 | - | 图像分类 |
| GPT-2 Small | ~117M | 12 | 768 | 文本生成 |
| GPT-2 Medium | ~345M | 24 | 1024 | 文本生成 |
| BERT-Base | ~110M | 12 | 768 | 文本理解 |
| BERT-Large | ~340M | 24 | 1024 | 文本理解 |

## 下一步

可以考虑的扩展：
1. 添加中文 BERT 支持（`bert-base-chinese`）
2. 添加 BERT 微调脚本
3. 添加 BERT 在特定任务上的压缩实验
4. 支持其他 BERT 变体（RoBERTa、ALBERT 等）

## 相关文档

- [docs/BERT_USAGE.md](../docs/BERT_USAGE.md) - BERT 使用指南
- [docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md) - 项目架构
- [docs/EXPERIMENTS.md](../docs/EXPERIMENTS.md) - 实验指南
- [CLAUDE.md](../CLAUDE.md) - 项目快速参考

## 验证清单

- [x] 创建 BERT 模型封装类
- [x] 添加 BERT-Base 和 BERT-Large 支持
- [x] 更新下载脚本
- [x] 更新模型模块初始化
- [x] 创建使用文档
- [x] 创建测试脚本
- [x] 创建示例脚本
- [x] 更新项目文档
- [x] 验证下载脚本工作正常

## 总结

成功为 `ckpt-compress` 项目添加了完整的 BERT-Large 模型支持，包括：
- 模型封装和加载
- 自动下载功能
- 完整的文档和示例
- 与现有架构的无缝集成

用户现在可以轻松地：
1. 下载 BERT-Large 预训练模型
2. 在代码中使用 BERT 模型
3. 将 BERT 模型用于检查点压缩实验
4. 参考文档和示例快速上手
