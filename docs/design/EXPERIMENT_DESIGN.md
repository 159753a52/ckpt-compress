# 实验设计方案

本文档规划项目的实验配置，支持 CV 和 NLP 任务的快速迭代和主结果实验。

## 实验配置矩阵

### CV（视觉）实验

| 配置 | 模型 | 数据集 | 用途 | 参数量 | 训练成本 |
|------|------|--------|------|--------|---------|
| **快速迭代** | ResNet-18 | CIFAR-10 | 调方法、sanity check、消融实验 | ~11M | 低 |
| **快速迭代+** | ResNet-18 | CIFAR-100 | 更多类别、更现实 | ~11M | 低 |
| **主结果（中等）** | ResNet-50 | CIFAR-100 | 性价比高、更现实 | ~25M | 中 |
| **主结果（中等+）** | ResNet-50 | Tiny-ImageNet | 更接近 ImageNet | ~25M | 中 |
| **主结果（最强）** | ResNet-50 | ImageNet | 最强说服力 | ~25M | 高 |

### NLP（语言）实验

| 配置 | 模型 | 数据集 | 用途 | 参数量 | 训练成本 |
|------|------|--------|------|--------|---------|
| **快速迭代** | GPT-2 Small | WikiText-2 | 快速验证、bits-ppl 曲线 | ~124M | 低 |
| **快速迭代+** | GPT-2 Small | WikiText-103 | 更大数据集 | ~124M | 中 |
| **主结果** | GPT-2 Medium | WikiText-103 | 中等规模、常用基准 | ~355M | 高 |

## 项目结构设计

```
ckpt-compress/
├── data/                           # 数据目录
│   ├── cifar10/                    # CIFAR-10 数据
│   ├── cifar100/                   # CIFAR-100 数据
│   ├── tiny-imagenet/              # Tiny-ImageNet 数据
│   ├── imagenet/                   # ImageNet 数据（需要手动下载）
│   ├── wikitext-2/                 # WikiText-2 数据
│   ├── wikitext-103/               # WikiText-103 数据
│   └── hf_cache/                   # HuggingFace 缓存
│
├── src/ckpt_compress/
│   ├── models/
│   │   ├── resnet.py               # ResNet-18/50 (CIFAR/ImageNet)
│   │   ├── gpt2.py                 # GPT-2 Small/Medium
│   │   └── __init__.py
│   │
│   ├── utils/
│   │   ├── data_loader.py          # 数据加载器（扩展）
│   │   ├── trainer.py              # 统一训练框架（新增）
│   │   └── checkpoint.py           # 检查点管理（新增）
│   │
│   └── methods/                    # 压缩方法（已有）
│
├── experiments/
│   ├── configs/                    # 实验配置文件（新增）
│   │   ├── cv/
│   │   │   ├── resnet18_cifar10.yaml
│   │   │   ├── resnet18_cifar100.yaml
│   │   │   ├── resnet50_cifar100.yaml
│   │   │   ├── resnet50_tiny_imagenet.yaml
│   │   │   └── resnet50_imagenet.yaml
│   │   └── nlp/
│   │       ├── gpt2_small_wikitext2.yaml
│   │       ├── gpt2_small_wikitext103.yaml
│   │       └── gpt2_medium_wikitext103.yaml
│   │
│   ├── scripts/
│   │   ├── train_cv.py             # CV 训练脚本（新增）
│   │   ├── train_nlp.py            # NLP 训练脚本（新增）
│   │   ├── compress_and_resume.py  # 压缩+恢复训练（新增）
│   │   └── download_data.py        # 数据下载脚本（新增）
│   │
│   └── results/                    # 实验结果
│
└── docs/
    ├── EXPERIMENT_DESIGN.md        # 本文档
    └── DATA_PREPARATION.md         # 数据准备指南（新增）
```

## 数据集详情

### CIFAR-10
- **大小**: 170 MB
- **样本数**: 60,000 (50,000 训练 + 10,000 测试)
- **类别数**: 10
- **图像尺寸**: 32x32
- **下载**: 自动（torchvision）

### CIFAR-100
- **大小**: 170 MB
- **样本数**: 60,000 (50,000 训练 + 10,000 测试)
- **类别数**: 100
- **图像尺寸**: 32x32
- **下载**: 自动（torchvision）

### Tiny-ImageNet
- **大小**: ~250 MB
- **样本数**: 120,000 (100,000 训练 + 10,000 验证 + 10,000 测试)
- **类别数**: 200
- **图像尺寸**: 64x64
- **下载**: 需要脚本（http://cs231n.stanford.edu/tiny-imagenet-200.zip）

### ImageNet (ILSVRC2012)
- **大小**: ~150 GB
- **样本数**: 1,431,167 (训练) + 50,000 (验证)
- **类别数**: 1000
- **图像尺寸**: 可变（通常 resize 到 224x224）
- **下载**: 需要手动下载（需要账号）

### WikiText-2
- **大小**: ~12 MB
- **Token 数**: ~2M
- **下载**: 自动（HuggingFace datasets）

### WikiText-103
- **大小**: ~500 MB
- **Token 数**: ~103M
- **下载**: 自动（HuggingFace datasets）

## 模型详情

### ResNet-18
- **参数量**: ~11M
- **层数**: 18
- **适用**: CIFAR-10/100, Tiny-ImageNet
- **修改**: CIFAR 需要修改第一层卷积和池化

### ResNet-50
- **参数量**: ~25M
- **层数**: 50
- **适用**: CIFAR-100, Tiny-ImageNet, ImageNet
- **修改**: CIFAR/Tiny-ImageNet 需要修改第一层

### GPT-2 Small
- **参数量**: ~124M
- **层数**: 12
- **隐藏维度**: 768
- **注意力头数**: 12
- **适用**: WikiText-2/103

### GPT-2 Medium
- **参数量**: ~355M
- **层数**: 24
- **隐藏维度**: 1024
- **注意力头数**: 16
- **适用**: WikiText-103

## 实验流程

### 快速迭代流程（ResNet-18 + CIFAR-10）

```bash
# 1. 训练基线模型
python experiments/scripts/train_cv.py \
  --config experiments/configs/cv/resnet18_cifar10.yaml \
  --epochs 200

# 2. 压缩检查点并恢复训练
python experiments/scripts/compress_and_resume.py \
  --checkpoint checkpoints/resnet18_cifar10_epoch100.pt \
  --method excp \
  --resume_epochs 50
```

### 主结果流程（ResNet-50 + CIFAR-100）

```bash
# 1. 训练基线模型
python experiments/scripts/train_cv.py \
  --config experiments/configs/cv/resnet50_cifar100.yaml \
  --epochs 200

# 2. 压缩检查点并恢复训练
python experiments/scripts/compress_and_resume.py \
  --checkpoint checkpoints/resnet50_cifar100_epoch100.pt \
  --method excp \
  --resume_epochs 100
```

### NLP 快速迭代流程（GPT-2 Small + WikiText-2）

```bash
# 1. 训练基线模型
python experiments/scripts/train_nlp.py \
  --config experiments/configs/nlp/gpt2_small_wikitext2.yaml \
  --epochs 10

# 2. 压缩检查点并恢复训练
python experiments/scripts/compress_and_resume.py \
  --checkpoint checkpoints/gpt2_small_wikitext2_epoch5.pt \
  --method predictive \
  --resume_epochs 5
```

## 评估指标

### CV 指标
- **准确率**: Top-1 和 Top-5 准确率
- **压缩率**: 压缩后大小 / 原始大小
- **恢复性能**: 恢复训练后的准确率
- **bits-acc 曲线**: 不同压缩率下的准确率

### NLP 指标
- **困惑度 (PPL)**: 语言模型性能
- **压缩率**: 压缩后大小 / 原始大小
- **恢复性能**: 恢复训练后的困惑度
- **bits-ppl 曲线**: 不同压缩率下的困惑度

## 硬件要求

| 实验配置 | 最低 GPU | 推荐 GPU | 显存 | 训练时间（估算） |
|---------|---------|---------|------|----------------|
| ResNet-18 + CIFAR-10 | GTX 1080 | RTX 3090 | 8GB | 2-4 小时 |
| ResNet-18 + CIFAR-100 | GTX 1080 | RTX 3090 | 8GB | 2-4 小时 |
| ResNet-50 + CIFAR-100 | RTX 2080 | RTX 3090 | 11GB | 4-8 小时 |
| ResNet-50 + Tiny-ImageNet | RTX 2080 | RTX 3090 | 11GB | 8-16 小时 |
| ResNet-50 + ImageNet | RTX 3090 | A100 | 24GB | 3-5 天 |
| GPT-2 Small + WikiText-2 | RTX 2080 | RTX 3090 | 11GB | 1-2 小时 |
| GPT-2 Small + WikiText-103 | RTX 3090 | A100 | 16GB | 8-16 小时 |
| GPT-2 Medium + WikiText-103 | A100 | A100 | 40GB | 1-2 天 |

## 下一步

1. 扩展模型模块（ResNet-50, GPT-2 Medium）
2. 扩展数据加载器（CIFAR-100, Tiny-ImageNet, WikiText-103）
3. 创建统一训练框架
4. 创建数据下载脚本
5. 创建实验配置文件
6. 创建压缩+恢复训练脚本
