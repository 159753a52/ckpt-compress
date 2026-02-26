# 数据准备指南

本文档说明如何下载和准备 `ckpt-compress` 项目所需的数据集和模型。

## 快速开始

### 自动下载所有数据和模型

```bash
# 下载所有数据集
python scripts/download_data.py --all

# 下载所有模型
python scripts/download_models.py --all
```

## 数据集

### CIFAR-10

**数据集信息**:
- 大小: ~170 MB
- 样本数: 60,000 (50,000 训练 + 10,000 测试)
- 类别数: 10
- 图像尺寸: 32x32x3

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_data.py --dataset cifar10

# 方法 2: 使用 Python API
python -c "from torchvision import datasets; datasets.CIFAR10(root='./data', download=True)"
```

**目录结构**:
```
data/
├── cifar-10-batches-py/
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── test_batch
│   └── batches.meta
└── cifar-10-python.tar.gz
```

---

### CIFAR-100

**数据集信息**:
- 大小: ~170 MB
- 样本数: 60,000 (50,000 训练 + 10,000 测试)
- 类别数: 100
- 图像尺寸: 32x32x3

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_data.py --dataset cifar100

# 方法 2: 使用 Python API
python -c "from torchvision import datasets; datasets.CIFAR100(root='./data', download=True)"
```

**目录结构**:
```
data/
├── cifar-100-python/
│   ├── train
│   ├── test
│   └── meta
└── cifar-100-python.tar.gz
```

---

### WikiText-2

**数据集信息**:
- 大小: ~11 MB
- Token 数: ~2M
- 适用于: GPT-2 Small 快速实验

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_data.py --dataset wikitext2

# 方法 2: 使用 Python API
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"
```

**目录结构**:
```
data/
└── hf_cache/
    └── wikitext/
        └── wikitext-2-raw-v1/
```

---

### WikiText-103

**数据集信息**:
- 大小: ~500 MB
- Token 数: ~103M
- 适用于: GPT-2 Medium 主结果实验

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_data.py --dataset wikitext103

# 方法 2: 使用 Python API
python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-103-raw-v1')"
```

**目录结构**:
```
data/
└── hf_cache/
    └── wikitext/
        └── wikitext-103-raw-v1/
```

---

## 预训练模型

### GPT-2 Small

**模型信息**:
- 参数量: ~124M
- 层数: 12
- 隐藏维度: 768

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_models.py --model gpt2-small

# 方法 2: 使用 Python API
python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2')"
```

**目录结构**:
```
data/models/
└── models--gpt2/
    └── snapshots/
        └── <hash>/
            ├── config.json
            ├── pytorch_model.bin
            └── ...
```

---

### GPT-2 Medium

**模型信息**:
- 参数量: ~355M
- 层数: 24
- 隐藏维度: 1024

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_models.py --model gpt2-medium

# 方法 2: 使用 Python API
python -c "from transformers import GPT2LMHeadModel; GPT2LMHeadModel.from_pretrained('gpt2-medium')"
```

**目录结构**:
```
data/models/
└── models--gpt2-medium/
    └── snapshots/
        └── <hash>/
            ├── config.json
            ├── pytorch_model.bin
            └── ...
```

---

### ResNet-50 (ImageNet 预训练)

**模型信息**:
- 参数量: ~25M
- 预训练数据集: ImageNet-1K
- 文件大小: ~98 MB

**下载方法**:

```bash
# 方法 1: 使用下载脚本
python scripts/download_models.py --model resnet50

# 方法 2: 使用 Python API
python -c "from torchvision import models; models.resnet50(pretrained=True)"
```

**目录结构**:
```
data/models/
└── checkpoints/
    └── resnet50-0676ba61.pth
```

---

## 数据验证

### 验证数据集是否已下载

```bash
# 验证单个数据集
python scripts/download_data.py --verify cifar10
python scripts/download_data.py --verify cifar100
python scripts/download_data.py --verify wikitext2
python scripts/download_data.py --verify wikitext103
```

### 验证数据完整性

```python
# 验证 CIFAR-10
from src.ckpt_compress.utils.data_loader import get_cifar10_loaders
train_loader, test_loader = get_cifar10_loaders(batch_size=128)
print(f"CIFAR-10 训练集: {len(train_loader.dataset)} 样本")
print(f"CIFAR-10 测试集: {len(test_loader.dataset)} 样本")

# 验证 WikiText-2
from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader
loader = get_wikitext2_dataloader(split='train', batch_size=8)
print(f"WikiText-2 训练集: {len(loader.dataset)} 样本")
```

---

## 存储空间要求

| 数据集/模型 | 大小 | 说明 |
|------------|------|------|
| CIFAR-10 | ~170 MB | 包含原始数据和压缩包 |
| CIFAR-100 | ~170 MB | 包含原始数据和压缩包 |
| WikiText-2 | ~11 MB | HuggingFace 缓存 |
| WikiText-103 | ~500 MB | HuggingFace 缓存 |
| GPT-2 Small | ~500 MB | 包含模型权重和配置 |
| GPT-2 Medium | ~1.5 GB | 包含模型权重和配置 |
| ResNet-50 | ~98 MB | 仅权重文件 |
| **总计** | **~3 GB** | 所有数据和模型 |

---

## 使用镜像源加速下载

### HuggingFace 镜像

如果 HuggingFace 下载速度慢，可以使用镜像源：

```bash
# 设置环境变量
export HF_ENDPOINT=https://hf-mirror.com

# 然后正常下载
python scripts/download_data.py --dataset wikitext2
python scripts/download_models.py --model gpt2-small
```

### ModelScope 镜像

```python
# 使用 ModelScope 下载 GPT-2
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/gpt2', cache_dir='./data/models')
```

---

## 常见问题

### Q: 下载速度很慢怎么办？

A: 可以尝试以下方法：
1. 使用 HuggingFace 镜像源（见上文）
2. 使用 ModelScope 镜像
3. 手动下载后放到指定目录

### Q: 磁盘空间不足怎么办？

A: 可以：
1. 只下载需要的数据集和模型
2. 删除下载后的压缩包（如 `.tar.gz` 文件）
3. 使用外部存储或网络存储

### Q: 如何使用本地已有的数据？

A: 将数据放到对应的目录结构中，或使用 `--data_dir` 参数指定路径：

```bash
# 使用本地 CIFAR-10
python experiments/scripts/train_cv.py --dataset cifar10 --data_dir /path/to/data

# 使用本地 WikiText-2
python experiments/scripts/train_nlp.py --dataset wikitext2 --data_dir /path/to/wikitext-2
```

### Q: 如何清理缓存？

A: 删除对应的目录：

```bash
# 清理 CIFAR 数据
rm -rf data/cifar-*

# 清理 HuggingFace 缓存
rm -rf data/hf_cache

# 清理模型缓存
rm -rf data/models
```

---

## 下一步

数据准备完成后，可以：

1. 查看 [EXPERIMENTS.md](EXPERIMENTS.md) 了解如何运行实验
2. 查看 [CLAUDE.md](../CLAUDE.md) 了解项目使用指南
3. 开始训练模型：
   ```bash
   # CV 训练
   python experiments/scripts/train_cv.py --model resnet18 --dataset cifar10

   # NLP 训练
   python experiments/scripts/train_nlp.py --model gpt2-small --dataset wikitext2
   ```
