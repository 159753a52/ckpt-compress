# 项目开发计划

本文档详细说明 `ckpt-compress` 项目的后续开发计划。

## 当前进度总结（2026-01-17 更新）

### ✅ 已完成

#### 1. 模型扩展
- **ResNet-50 支持** ✅
  - `ResNet50ForCIFAR`: 支持 CIFAR-10/100（32x32 图像）
  - `ResNet50ForImageNet`: 支持 ImageNet 和 Tiny-ImageNet（224x224 和 64x64 图像）
  - 工厂函数：`get_resnet50_cifar100()`, `get_resnet50_tiny_imagenet()`, `get_resnet50_imagenet()`
  - 测试：38 个测试用例，覆盖率 93.48% ✅

- **GPT-2 Medium 支持** ✅
  - `GPT2MediumForExperiment`: 355M 参数，24 层，1024 隐藏维度
  - 工厂函数：`get_gpt2_medium()`
  - 测试：28 个测试用例（包含 GPT-2 Small 和 Medium）
  - 测试状态：全部通过，覆盖率 **100%** ✅（已达标！）

#### 2. 数据加载器扩展
- **CIFAR-100 支持** ✅
  - `get_cifar100_loaders()`: 完整实现 ✅
  - 支持数据增强、子集采样
  - 测试：54 个测试用例 ✅

- **WikiText-103 支持** ✅
  - `get_wikitext103_dataloader()`: 完整实现 ✅
  - 支持本地加载和 HuggingFace 下载
  - 复用 WikiText2Dataset 类
  - 测试：已包含在 54 个测试用例中 ✅

#### 3. 统一训练框架
- **BaseTrainer** ✅
  - 完整实现训练循环、验证循环
  - 支持检查点保存/加载
  - 支持早停机制
  - 支持学习率调度
  - 支持自动混合精度训练（AMP）
  - 支持梯度累积
  - 测试：20 个测试用例 ✅

#### 4. 数据和模型下载脚本
- **scripts/download_data.py** ✅
  - 支持 CIFAR-10、CIFAR-100、WikiText-2、WikiText-103
  - 支持数据验证功能
  - 命令行接口完整
  - 测试：已实现 ✅

- **scripts/download_models.py** ✅
  - 支持 GPT-2 Small、GPT-2 Medium、ResNet-50
  - 命令行接口完整
  - 测试：已实现 ✅

#### 5. 文档更新
- 精简 CLAUDE.md（从 482 行 → 197 行）✅
- 创建 docs/ARCHITECTURE.md（详细架构文档）✅
- 创建 docs/EXPERIMENTS.md（实验指南）✅
- 创建 docs/EXPERIMENT_DESIGN.md（实验设计方案）✅

---

## 待完成任务（按优先级排序）

### ~~🔴 优先级 1: 提升测试覆盖率到 90%~~ ✅ 已完成

#### ~~任务 1.1: 提升 GPT-2 模型测试覆盖率~~ ✅ 已完成
**状态**: ✅ 已完成 - 覆盖率达到 100%

#### ~~任务 1.2: 为 CIFAR-100 数据加载器编写测试~~ ✅ 已完成
**状态**: ✅ 已完成 - 54 个测试用例已实现，包含 CIFAR-100 和 WikiText-103 测试

---

### ~~🟡 优先级 2: 数据加载器扩展~~ ✅ 已完成

#### ~~任务 2.1: WikiText-103 数据加载器扩展~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现功能**:
- ✅ `get_wikitext103_dataloader()` 函数完整实现（data_loader.py:334-429）
- ✅ 支持 HuggingFace 下载和本地文件加载
- ✅ 复用 WikiText2Dataset 类
- ✅ 支持增量加载和内存效率优化
- ✅ 测试已包含在 54 个测试用例中

---

#### ~~任务 2.2: Tiny-ImageNet 数据加载器扩展~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现** (src/ckpt_compress/utils/data_loader.py):
- ✅ `TinyImageNetDataset` 类（data_loader.py:432-693）
  - 支持训练集和验证集
  - 支持自动下载（从 Stanford CS231n）
  - 完整的错误处理
  - 200 个类别，64x64 图像
- ✅ `get_tiny_imagenet_transforms()` 函数
- ✅ `get_tiny_imagenet_loaders()` 函数
- ✅ 测试：28 个测试用例（tests/unit/utils/test_tiny_imagenet.py）
- ✅ 覆盖率：**99.05%** ✅（远超 90% 要求）

---

### ~~🟢 优先级 3: 数据下载和管理~~ ✅ 已完成

#### ~~任务 3.1: 数据下载脚本~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现功能** (scripts/download_data.py):
- ✅ `download_cifar10()` - 下载 CIFAR-10
- ✅ `download_cifar100()` - 下载 CIFAR-100
- ✅ `download_wikitext2()` - 下载 WikiText-2
- ✅ `download_wikitext103()` - 下载 WikiText-103
- ✅ `download_all()` - 批量下载数据集
- ✅ `verify_dataset()` - 验证数据集完整性
- ✅ 完整的命令行接口（argparse）
- ✅ 错误处理和状态报告
- ✅ 测试已实现（tests/unit/scripts/test_download_data.py）

---

#### ~~任务 3.2: 模型权重下载脚本~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现功能** (scripts/download_models.py):
- ✅ `download_gpt2_small()` - 下载 GPT-2 Small
- ✅ `download_gpt2_medium()` - 下载 GPT-2 Medium
- ✅ `download_resnet50_imagenet()` - 下载 ResNet-50
- ✅ `download_all_models()` - 批量下载模型
- ✅ 完整的命令行接口（argparse）
- ✅ 错误处理和状态报告
- ✅ 测试已实现（tests/unit/scripts/test_download_models.py）

---

### ~~🔵 优先级 4: 统一训练框架~~ ✅ 已完成

#### ~~任务 4.1: 训练器基类~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现** (src/ckpt_compress/utils/trainer.py):
- ✅ `BaseTrainer` 类完整实现（trainer.py:17-348）
  - ✅ 训练循环 (`train_one_epoch()`)
  - ✅ 验证循环 (`validate()`)
  - ✅ 检查点保存/加载 (`save_checkpoint()`, `load_checkpoint()`)
  - ✅ 早停机制
  - ✅ 学习率调度支持
  - ✅ 自动混合精度训练（AMP）
  - ✅ 梯度累积
  - ✅ 支持 CV 和 NLP 任务（通过灵活的批次处理）
- ✅ 测试：20 个测试用例（tests/unit/utils/test_trainer.py）

**说明**: BaseTrainer 已经通过灵活的批次处理逻辑同时支持 CV 和 NLP 任务，因此不需要单独的子类。

---

#### ~~任务 4.2: 训练脚本~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现**:
- ✅ `experiments/scripts/train_cv.py` - CV 模型训练脚本
  - 支持 ResNet18/ResNet50
  - 支持 CIFAR-10/CIFAR-100
  - 完整的命令行参数
  - 支持学习率调度、早停、AMP 等功能

- ✅ `experiments/scripts/train_nlp.py` - NLP 模型训练脚本
  - 支持 GPT-2 Small/Medium
  - 支持 WikiText-2/WikiText-103
  - 完整的命令行参数
  - 支持预训练权重加载

---

#### ~~任务 4.3: 压缩和恢复训练脚本~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现**:
- ✅ `experiments/scripts/compress_and_resume.py`
  - 支持三种压缩方法：ExCP, Inshrinkerator, PredictiveResidual
  - 压缩模式：压缩检查点并显示统计信息
  - 解压模式：从压缩检查点恢复
  - 保留优化器状态和训练历史

---

### ~~🟣 优先级 5: 实验配置文件~~ ✅ 已完成

#### ~~任务 5.1: YAML 配置文件~~ ✅ 已完成
**状态**: ✅ 已完成

**已实现**:
- ✅ `experiments/configs/cv/resnet18_cifar10.yaml`
- ✅ `experiments/configs/cv/resnet18_cifar100.yaml`
- ✅ `experiments/configs/cv/resnet50_cifar100.yaml`
- ✅ `experiments/configs/nlp/gpt2_small_wikitext2.yaml`
- ✅ `experiments/configs/nlp/gpt2_small_wikitext103.yaml`
- ✅ `experiments/configs/nlp/gpt2_medium_wikitext103.yaml`

所有配置文件包含完整的模型、数据集、训练和检查点参数。

---

### ~~📝 优先级 6: 文档更新~~ ✅ 已完成

#### ~~任务 6.1: 更新 CLAUDE.md~~ ✅ 已完成
**状态**: ✅ 已完成

**已更新内容**:
- ✅ 添加数据和模型下载命令
- ✅ 添加训练脚本使用示例
- ✅ 添加检查点压缩命令
- ✅ 添加数据加载器说明
- ✅ 添加统一训练框架说明
- ✅ 更新目录结构（包含 trainer.py）
- ✅ 添加新文档链接

#### ~~任务 6.2: 更新 EXPERIMENTS.md~~ ✅ 已完成
**状态**: ✅ 已完成

**已更新内容**:
- ✅ 添加数据准备章节
- ✅ 添加训练脚本使用示例（CV 和 NLP）
- ✅ 添加检查点压缩和解压示例
- ✅ 添加配置文件使用说明
- ✅ 保留原有的 AdamPrune 实验说明

#### ~~任务 6.3: 创建 DATA_PREPARATION.md~~ ✅ 已完成
**状态**: ✅ 已完成

**已创建内容**:
- ✅ 所有数据集的下载方法和目录结构
  - CIFAR-10/100
  - WikiText-2/103
- ✅ 所有模型的下载方法和目录结构
  - GPT-2 Small/Medium
  - ResNet-50
- ✅ 数据验证方法
- ✅ 存储空间要求
- ✅ 镜像源加速下载说明
- ✅ 常见问题解答

---

## 开发原则

### TDD（测试驱动开发）
1. 先编写测试
2. 运行测试（应该失败）
3. 编写最小代码使测试通过
4. 重构代码
5. 重复

### 代码质量要求
- 遵循 PEP 8 规范
- 使用 black 格式化
- 使用 isort 排序导入
- 使用 mypy 类型检查
- **代码覆盖率 ≥90%**（强制要求）

### 简洁高效原则
- 避免过度工程
- 保持代码简单
- 优先使用现有工具和库
- 避免不必要的抽象
- 不添加未来可能需要的功能

---

## 测试覆盖率目标

所有新增代码必须达到 **90%** 的测试覆盖率。

### 当前覆盖率状态
- ResNet 模型：93.48% ✅
- GPT-2 模型：100% ✅
- Data Loader：97% ✅
- Trainer：覆盖率良好 ✅

### 覆盖率验证命令
```bash
# 运行所有测试并生成覆盖率报告
pytest --cov=src/ckpt_compress --cov-report=html --cov-report=term-missing

# 查看 HTML 报告
open htmlcov/index.html

# 检查特定模块
pytest tests/unit/models/test_gpt2.py --cov=ckpt_compress.models.gpt2 --cov-report=term-missing
```

---

## 注意事项

### 内存管理
- GPT-2 Medium 模型很大（355M 参数），测试时注意内存使用
- 使用小批次大小进行测试
- 考虑使用 pytest 标记跳过慢速测试

### 数据下载
- 某些数据集（如 ImageNet）需要手动下载
- 测试时使用 mock 避免实际下载
- 提供本地数据路径选项

### GPU 要求
- 某些实验需要 GPU 支持
- 训练脚本应支持 CPU 模式（用于测试）

### 测试时间
- 大模型测试可能需要较长时间
- 使用 `@pytest.mark.slow` 标记慢速测试
- 提供快速测试选项

---

## 后续扩展（可选）

### ImageNet 支持
- 需要手动下载（需要账号）
- 数据预处理脚本
- 分布式训练支持

### 更多压缩方法
- 新的压缩算法实现
- 压缩方法对比实验
- 性能优化

### 可视化工具
- 训练曲线可视化
- 压缩率 vs 性能曲线
- 模型参数分布可视化

---

## 执行顺序建议

### ~~第一阶段（本周）~~ ✅ 已完成
1. ✅ 提升 GPT-2 测试覆盖率到 90%
2. ✅ 为 CIFAR-100 编写测试
3. ✅ 实现 WikiText-103 数据加载器
4. ✅ 为 WikiText-103 编写测试

### ~~第二阶段（下周）~~ ✅ 已完成
1. ✅ 创建数据下载脚本
2. ✅ 创建模型权重下载脚本
3. ✅ 开始训练框架开发

### ~~第三阶段（两周后）~~ ✅ 已完成
1. ✅ 完成统一训练框架
2. ✅ 创建训练脚本
3. ✅ 创建压缩和恢复训练脚本
4. ✅ 创建实验配置文件

### ~~第四阶段（三周后）~~ ✅ 已完成
1. ✅ 更新所有文档
2. ✅ 下载所有数据和模型
3. ✅ 准备运行端到端实验

---

## 🎉 项目状态总结

### 已完成的主要功能

**核心功能**:
- ✅ 4 种压缩方法（ExCP, Inshrinkerator, PredictiveResidual, AdamPrune）
- ✅ 统一训练框架（BaseTrainer）
- ✅ 完整的数据加载器（CIFAR-10/100, WikiText-2/103, Tiny-ImageNet）
- ✅ 模型支持（ResNet18/50, GPT-2 Small/Medium）

**工具脚本**:
- ✅ 数据下载脚本（download_data.py）
- ✅ 模型下载脚本（download_models.py）
- ✅ CV 训练脚本（train_cv.py）
- ✅ NLP 训练脚本（train_nlp.py）
- ✅ 检查点压缩脚本（compress_and_resume.py）

**配置和文档**:
- ✅ 6 个 YAML 配置文件
- ✅ 完整的文档体系（CLAUDE.md, EXPERIMENTS.md, DATA_PREPARATION.md, ARCHITECTURE.md, PLAN.md）
- ✅ 测试覆盖率 ≥90%

**数据和模型**:
- ✅ 所有数据集已下载（~889 MB）
- ✅ 所有模型已下载（~468 MB）

### 下一步建议

1. **运行端到端实验**: 使用新的训练脚本训练模型并测试压缩
2. **性能优化**: 根据实验结果优化压缩方法
3. **论文实验**: 运行完整的对比实验生成论文数据
4. **可选扩展**:
   - ~~Tiny-ImageNet 支持~~ ✅ 已完成
   - 更多压缩方法
   - 可视化工具
   - ImageNet 支持

---

## 联系和反馈

如有问题或建议，请在项目 issue 中提出。
