# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的快速参考指导。

## 项目概述

`ckpt-compress` 是一个基于 PyTorch 的深度学习模型检查点压缩方法对比研究框架。

**核心压缩方法**:
- **ExCP**: 残差编码 + 权重-动量联合剪枝 + K-means 量化
- **Inshrinkerator**: 三向分区 + DDSketch 近似 K-means + RLE 增量编码
- **PredictiveResidual**: Adam 权重预测 + 敏感度自适应量化 + 优化器状态压缩
- **AdamPrune**: 基于 Adam 二阶矩的理论剪枝方法（重要性得分 + 分布拟合 + 幂律校正）

**技术栈**: PyTorch, NumPy, SciPy

## 快速开始

```bash
# 安装项目（含开发依赖）
pip install -e ".[dev]"

# 运行测试
pytest

# 运行主要实验（GPT-2 + WikiText-2）
python experiments/scripts/run_formula_comparison.py --mode comparison --max_samples 100
```

## 常用命令

### 测试

```bash
pytest                                    # 运行所有测试
pytest tests/unit/methods/test_excp_e2e.py  # 运行指定测试
pytest --cov=src/ckpt_compress --cov-report=html  # 生成覆盖率报告
pytest -m "not slow"                      # 跳过慢速测试
```

### 代码质量

```bash
black src/ tests/      # 格式化代码
isort src/ tests/      # 排序导入
mypy src/ckpt_compress # 类型检查
```

### 数据和模型下载

```bash
# 下载所有数据集
python scripts/download_data.py --all

# 下载所有模型
python scripts/download_models.py --all

# 下载单个数据集
python scripts/download_data.py --dataset cifar10
python scripts/download_data.py --dataset wikitext103

# 下载单个模型
python scripts/download_models.py --model gpt2-medium
python scripts/download_models.py --model bert-large

# 下载多个模型
python scripts/download_models.py --model bert-base,bert-large
```

详见 [docs/guides/DATA_PREPARATION.md](docs/guides/DATA_PREPARATION.md) 和 [docs/guides/BERT_USAGE.md](docs/guides/BERT_USAGE.md)

### 训练脚本

```bash
# CV 训练（ResNet18 on CIFAR-10）
python experiments/scripts/finetune/train_cv.py \
  --model resnet18 \
  --dataset cifar10 \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.1

# NLP 训练（GPT-2 Small on WikiText-2）
python experiments/scripts/finetune/train_nlp.py \
  --model gpt2-small \
  --dataset wikitext2 \
  --epochs 10 \
  --batch_size 8 \
  --lr 5e-5

# 使用配置文件训练
python experiments/scripts/finetune/train_cv.py --config experiments/configs/cv/resnet18_cifar10.yaml
```

### 检查点压缩

```bash
# 压缩检查点
python experiments/scripts/compress/compress_and_resume.py \
  --mode compress \
  --checkpoint checkpoints/model.pt \
  --method excp \
  --output compressed/model_excp.bin

# 解压检查点
python experiments/scripts/compress/compress_and_resume.py \
  --mode decompress \
  --compressed compressed/model_excp.bin \
  --output checkpoints/model_restored.pt
```

### AdamPrune 实验脚本

```bash
# AdamPrune: 公式对比实验
python experiments/scripts/run_formula_comparison.py --mode comparison --max_samples 100

# AdamPrune: 幂律校正实验
python experiments/scripts/run_formula_comparison.py --mode calibration --max_samples 100

# AdamPrune: 自适应剪枝实验
python experiments/scripts/run_adaptive_pruning.py --max_samples 100

# 压缩方法对比（ResNet18 + CIFAR-10）
python experiments/scripts/run_comparison.py
```

更多实验脚本详见 [docs/guides/EXPERIMENTS.md](docs/guides/EXPERIMENTS.md)

## 核心架构

### 目录结构

```
src/ckpt_compress/
├── core/
│   └── base.py              # BaseCompressor 抽象基类
├── methods/
│   ├── excp/                # ExCP 压缩方法
│   ├── inshrinkerator/      # Inshrinkerator 压缩方法
│   ├── predictive/          # PredictiveResidual 压缩方法
│   └── adam_prune/          # AdamPrune 理论剪枝方法
├── utils/
│   ├── tensor_ops.py        # 张量操作工具
│   ├── optimizer_utils.py   # 优化器工具
│   ├── data_loader.py       # 数据加载器
│   └── trainer.py           # 统一训练框架
└── models/
    ├── resnet.py            # ResNet 模型（ResNet18/50）
    ├── gpt2.py              # GPT-2 封装（Small/Medium）
    └── bert.py              # BERT 封装（Base/Large）
```

### BaseCompressor 抽象类

所有压缩方法都继承自 `BaseCompressor`：

```python
class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """压缩模型状态字典"""

    @abstractmethod
    def decompress(self, data: bytes) -> Dict[str, torch.Tensor]:
        """解压字节数据恢复状态字典"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回方法名称"""
```

详细架构说明见 [docs/design/ARCHITECTURE.md](docs/design/ARCHITECTURE.md)

## 开发约定

### 添加新的压缩方法

1. 在 `methods/<method_name>/` 创建目录
2. 继承 `BaseCompressor` 并实现 `compress()`, `decompress()`, `name`
3. 添加端到端测试 `tests/unit/methods/test_<method_name>_e2e.py`
4. 更新 `src/ckpt_compress/methods/__init__.py` 导出

### 测试约定

- 端到端测试命名为 `test_<method>_e2e.py`
- 单元测试按模块命名，如 `test_<method>_<module>.py`
- 使用 `conftest.py` 中的共享 fixtures

### 代码风格

- 使用 black 格式化（line-length=88）
- 使用 isort 排序导入（profile="black"）
- 使用 mypy 进行类型检查

## 关键注意事项

### ✅ 应该做

- **大模型实验**: 使用 `memory_efficient.py` 模块避免 OOM
- **添加新方法**: 继承 `BaseCompressor` 作为基类
- **修改代码前**: 先使用 Read 工具读取文件
- **内存控制**: 使用 `--max_samples` 限制数据集大小

### ❌ 不要做

- **不要使用 deepcopy**: 大模型评估时使用 `MemoryEfficientEvaluator`
- **不要剪枝 bias**: AdamPrune 默认保留 bias 参数
- **不要跳过测试**: 添加新方法时必须同步更新测试
- **不要忽略内存**: GPT-2 实验建议 32GB+ 内存

### 内存管理（GPT-2 实验）

```bash
# 推荐配置
python experiments/scripts/run_formula_comparison.py \
  --max_samples 100 \          # 限制数据集大小
  --num_batches 10 \           # 缓存批次数
  --batch_size 4 \             # 批次大小
  --memory_limit_gb 30         # 内存限制
```

**关键策略**:
- 使用 `MemoryEfficientEvaluator` 进行原地修改 + 恢复（避免 deepcopy）
- 使用 `cache_batches()` 确保梯度计算和损失评估使用相同数据
- 可选 CPU 卸载策略减少 GPU 显存占用

### AdamPrune 重要性得分

支持两种计算方法：

1. **Adam 近似（默认）**: `s_i = -g_i * θ_i + α * v_i * θ_i²`
   - 快速，但只考虑对角元素

2. **HVP 精确**: `s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i`
   - 准确，但计算慢，需要两次反向传播
   - 大模型使用 `compute_importance_scores_hvp_memory_efficient()`

## 实验结果

实验结果保存在 `results/` 目录，按类型组织：
- `results/paper_results/`: 论文核心结果
- `results/supplementary/`: 补充材料（可视化等）
- `results/archive/`: 归档的探索性实验

## 已知问题

- 部分代码存在 Pyright 类型检查警告（numpy/scipy 类型推断问题）
- `delta_encoding.py` 和 `quantization.py` 中存在 float 转 int 的类型警告
- AdamPrune 分布拟合可能失败（返回 None），使用前需检查

## 数据加载器

### 支持的数据集

**计算机视觉**:
- CIFAR-10: `get_cifar10_loaders()`
- CIFAR-100: `get_cifar100_loaders()`

**自然语言处理**:
- WikiText-2: `get_wikitext2_dataloader()`
- WikiText-103: `get_wikitext103_dataloader()`

### 使用示例

```python
# CIFAR-10
from src.ckpt_compress.utils.data_loader import get_cifar10_loaders
train_loader, test_loader = get_cifar10_loaders(batch_size=128)

# WikiText-2
from src.ckpt_compress.utils.data_loader import get_wikitext2_dataloader
loader = get_wikitext2_dataloader(split='train', batch_size=8, seq_length=512)
```

## 统一训练框架

`BaseTrainer` 提供统一的训练接口，支持：
- 训练和验证循环
- 检查点保存/加载
- 早停机制
- 学习率调度
- 自动混合精度训练（AMP）
- 梯度累积

```python
from src.ckpt_compress.utils.trainer import BaseTrainer

trainer = BaseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device='cuda',
    checkpoint_dir='./checkpoints',
    early_stopping_patience=5,
    use_amp=True,
)

history = trainer.train(epochs=100, save_every=10)
```

## 数据路径

- 默认数据从 HuggingFace/torchvision 自动下载
- 使用 `--data_dir` 指定本地数据路径
- 缓存目录：
  - CIFAR: `./data/`
  - WikiText: `./data/hf_cache/`
  - 模型: `./data/models/`

## 更多文档

- [docs/design/ARCHITECTURE.md](docs/design/ARCHITECTURE.md) - 详细架构设计
- [docs/guides/EXPERIMENTS.md](docs/guides/EXPERIMENTS.md) - 实验指南
- [docs/guides/DATA_PREPARATION.md](docs/guides/DATA_PREPARATION.md) - 数据准备指南
- [docs/design/PLAN.md](docs/design/PLAN.md) - 开发计划

## 项目目录结构

```
ckpt-compress/
├── CLAUDE.md                        # 快速参考
├── pyproject.toml                   # 项目配置
├── paper/                           # 论文
├── plan/                            # 规划文档
├── src/ckpt_compress/               # 源代码
├── tests/                           # 测试
├── experiments/
│   ├── configs/
│   │   ├── cv/                      # CV 训练配置
│   │   ├── nlp/                     # NLP 训练配置
│   │   └── pruning/                 # 剪枝配置
│   └── scripts/
│       ├── finetune/                # 微调脚本
│       ├── compress/                # 压缩脚本
│       ├── prune/                   # 剪枝脚本
│       ├── analysis/                # 分析脚本
│       ├── config/                  # 配置工具
│       └── comparison/              # 对比实验
├── results/
│   ├── paper_results/               # 论文核心结果
│   ├── supplementary/               # 补充材料
│   └── archive/                     # 归档实验
├── docs/
│   ├── guides/                      # 使用指南
│   ├── design/                      # 设计文档
│   ├── reference/                   # 参考文档
│   └── archive/                     # 归档文档
├── checkpoints/                     # 模型检查点
├── data/                            # 数据集
└── archive/                         # 历史归档
```
