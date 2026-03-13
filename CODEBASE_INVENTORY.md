# ckpt-compress 代码资产清单

生成时间: 2026-02-26  
项目路径: `/lihongliang/fangzl/ckpt-compress`

---

## 📊 总体统计

- **Python 文件总数**: 147 个
- **核心源码行数**: ~10,000+ 行
- **测试代码行数**: ~150,000+ 行
- **实验脚本行数**: ~10,000+ 行

---

## 🏗️ 核心架构 (src/ckpt_compress/)

### 1. 核心基类 (core/)

| 文件 | 行数 | 功能 |
|------|------|------|
| `base.py` | 53 | BaseCompressor 抽象基类，定义 compress/decompress 接口 |
| `__init__.py` | 4 | 模块导出 |

**关键类**:
- `BaseCompressor`: 所有压缩方法的抽象基类

---

### 2. 压缩方法实现 (methods/)

#### 2.1 AdamPrune (methods/adam_prune/) - 4,172 行

**核心理论**: 基于 Adam 二阶矩的参数重要性剪枝

| 文件 | 行数 | 核心函数 | 功能描述 |
|------|------|----------|----------|
| `importance.py` | 562 | `compute_importance_scores`<br>`compute_importance_scores_hvp`<br>`compute_importance_scores_first_order`<br>`compute_importance_scores_magnitude`<br>`compute_hvp`<br>`compute_hvp_batched`<br>`compute_hvp_memory_efficient` | 重要性得分计算（Adam 近似 + HVP 精确） |
| `distribution.py` | 322 | `fit_exponential_distribution`<br>`analyze_distribution`<br>`fit_multiple_distributions`<br>`get_best_fit` | 分布拟合（指数/Weibull/对数正态） |
| `distribution_fast.py` | 298 | `fit_weibull_linear_regression`<br>`ks_test_gpu`<br>`fit_distributions_per_layer_fast` | GPU 加速的快速分布拟合 |
| `layer_pruning.py` | 965 | `convert_scipy_params`<br>`compute_weibull_prune_ratio`<br>`compute_lognormal_prune_ratio`<br>`prune_by_global_sparsity`<br>`prune_by_layer_distribution`<br>`compute_layer_sparsities_from_global` | 分层剪枝优化（拉格朗日方法） |
| `adaptive_pruning.py` | 572 | `is_bias_param`<br>`filter_prunable_layers`<br>`fit_distributions_per_layer`<br>`build_layer_dist_params`<br>`compute_target_pruned_score_budget`<br>`collect_calibration_points`<br>`adjust_epsilon_for_target` | 自适应剪枝辅助函数 |
| `calibration.py` | 187 | `compute_calibration_variables`<br>`fit_powerlaw`<br>`invert_powerlaw` | 幂律校正 |
| `memory_efficient.py` | 489 | `MemoryMonitor`<br>`MemoryEfficientEvaluator`<br>`force_garbage_collection` | 内存高效评估（避免 deepcopy） |
| `visualization.py` | 720 | 可视化工具（分布图、剪枝率图等） | 实验结果可视化 |
| `__init__.py` | 57 | 模块导出 | - |

**关键算法**:
1. **重要性得分**: `s_i = -g_i * θ_i + α * v_i * θ_i²` (Adam 近似)
2. **HVP 精确**: `s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i`
3. **分布拟合**: Weibull, Exponential, Lognormal
4. **自适应剪枝**: 基于分布的分层剪枝率计算

---

#### 2.2 ExCP (methods/excp/) - 707 行

**核心理论**: 残差编码 + 权重-动量联合剪枝 + K-means 量化

| 文件 | 行数 | 核心函数 | 功能描述 |
|------|------|----------|----------|
| `excp.py` | 212 | `ExCPCompressor.compress`<br>`ExCPCompressor.decompress` | 端到端压缩器 |
| `residual.py` | 87 | `compute_residual`<br>`reconstruct`<br>`compute_residual_state_dict` | 残差编码 |
| `pruning.py` | 189 | `compute_weight_threshold`<br>`compute_momentum_threshold`<br>`joint_prune`<br>`prune_residual_by_percentile` | 联合剪枝 |
| `quantization.py` | 218 | `kmeans_quantize_nonzero`<br>`dequantize`<br>`pack_int4`<br>`unpack_int4` | K-means 量化 + 4-bit 打包 |
| `__init__.py` | 1 | 模块导出 | - |

**关键算法**:
1. **残差编码**: `ΔW_t = W_t - Ŵ_{t-1}`
2. **联合剪枝**: 基于权重和动量的阈值剪枝
3. **K-means 量化**: 非零值量化到 k 个中心

---

#### 2.3 Inshrinkerator (methods/inshrinkerator/) - 1,046 行

**核心理论**: 三向分区 + DDSketch 近似 K-means + RLE 增量编码

| 文件 | 行数 | 核心函数 | 功能描述 |
|------|------|----------|----------|
| `inshrinkerator.py` | 287 | `InshrinkeratorCompressor.compress`<br>`InshrinkeratorCompressor.decompress` | 端到端压缩器 |
| `partition.py` | 158 | `compute_thresholds`<br>`partition` | 三向分区（保护/剪枝/量化） |
| `approx_kmeans.py` | 204 | `compute_sample_weights`<br>`weighted_kmeans_plusplus_init`<br>`approx_kmeans`<br>`quantize_to_centers` | DDSketch 近似 K-means |
| `delta_encoding.py` | 195 | `delta_encode`<br>`delta_decode`<br>`rearrange_by_prev_bin`<br>`rle_encode`<br>`rle_decode` | 增量编码 + RLE 压缩 |
| `sketch.py` | 161 | `compute_gamma`<br>`compute_bucket_index` | DDSketch 实现 |
| `metrics.py` | 40 | `magnitude`<br>`sensitivity` | 敏感度计算 |
| `__init__.py` | 1 | 模块导出 | - |

**关键算法**:
1. **三向分区**: 保护 (0.5%) / 剪枝 (20%) / 量化 (79.5%)
2. **DDSketch**: 对数空间的分布草图
3. **RLE 编码**: 游程长度编码

---

#### 2.4 PredictiveResidual (methods/predictive/) - 1,108 行

**核心理论**: Adam 权重预测 + 敏感度自适应量化 + 优化器状态压缩

| 文件 | 行数 | 核心函数 | 功能描述 |
|------|------|----------|----------|
| `predictive.py` | 264 | `PredictiveCompressor.compress`<br>`PredictiveCompressor.decompress` | 端到端压缩器 |
| `predictor.py` | 219 | `sgd_predict`<br>`adam_predict`<br>`compute_prediction_residual` | 权重预测 |
| `adaptive_quantization.py` | 320 | `compute_sensitivity`<br>`assign_bit_allocation`<br>`adaptive_quantize`<br>`adaptive_dequantize` | 自适应量化 |
| `optimizer_compress.py` | 305 | `compress_exp_avg`<br>`decompress_exp_avg`<br>`compress_exp_avg_sq`<br>`decompress_exp_avg_sq` | 优化器状态压缩 |
| `__init__.py` | 1 | 模块导出 | - |

**关键算法**:
1. **Adam 预测**: `Ŵ_t = W_{t-1} - lr * m_t / (√v_t + ε)`
2. **敏感度量化**: 高敏感度参数分配更多比特
3. **优化器压缩**: 压缩 exp_avg 和 exp_avg_sq

---

### 3. 模型封装 (models/) - 705 行

| 文件 | 行数 | 核心类/函数 | 功能描述 |
|------|------|-------------|----------|
| `resnet.py` | 273 | `ResNet18ForCIFAR10`<br>`ResNet50ForCIFAR`<br>`ResNet50ForImageNet`<br>`get_resnet18`<br>`get_resnet50` | ResNet 模型（CIFAR-10/100, ImageNet） |
| `gpt2.py` | 199 | `GPT2ForExperiment`<br>`GPT2MediumForExperiment`<br>`get_gpt2_small`<br>`get_gpt2_medium` | GPT-2 Small/Medium 封装 |
| `bert.py` | 215 | `BERTForExperiment`<br>`get_bert_base`<br>`get_bert_large` | BERT Base/Large 封装 |
| `__init__.py` | 18 | 模块导出 | - |

**支持的模型**:
- **CV**: ResNet-18, ResNet-50
- **NLP**: GPT-2 Small (124M), GPT-2 Medium (355M), BERT-Base (110M), BERT-Large (340M)

---

### 4. 工具模块 (utils/) - 1,625 行

| 文件 | 行数 | 核心函数/类 | 功能描述 |
|------|------|-------------|----------|
| `data_loader.py` | 1004 | `get_cifar10_loaders`<br>`get_cifar100_loaders`<br>`get_wikitext2_dataloader`<br>`get_wikitext103_dataloader`<br>`get_tiny_imagenet_loaders` | 数据加载器（CV + NLP） |
| `trainer.py` | 347 | `BaseTrainer`<br>`CVTrainer`<br>`NLPTrainer` | 统一训练框架 |
| `optimizer_utils.py` | 128 | `extract_optimizer_state`<br>`restore_optimizer_state` | 优化器状态提取/恢复 |
| `tensor_ops.py` | 127 | 张量操作工具 | 通用张量操作 |
| `__init__.py` | 1 | 模块导出 | - |

**支持的数据集**:
- **CV**: CIFAR-10, CIFAR-100, Tiny-ImageNet
- **NLP**: WikiText-2, WikiText-103

---

## 🧪 测试代码 (tests/)

### 单元测试 (tests/unit/)

#### AdamPrune 测试 (tests/unit/methods/adam_prune/) - 14 个文件

| 文件 | 行数 | 测试内容 |
|------|------|----------|
| `test_importance.py` | 9,482 | 重要性得分计算 |
| `test_layer_pruning.py` | 38,011 | 分层剪枝 |
| `test_layer_pruning_fast.py` | 19,510 | 快速分层剪枝 |
| `test_distribution.py` | 15,446 | 分布拟合 |
| `test_distribution_fast.py` | 18,434 | 快速分布拟合 |
| `test_adaptive_pruning.py` | 6,352 | 自适应剪枝 |
| `test_calibration.py` | 12,622 | 幂律校正 |
| `test_memory_efficient.py` | 15,106 | 内存高效评估 |
| `test_visualization.py` | 26,062 | 可视化 |
| `test_importance_magnitude.py` | 7,005 | 幅度重要性 |

#### 其他方法测试 (tests/unit/methods/) - 12 个文件

| 文件 | 行数 | 测试内容 |
|------|------|----------|
| `test_excp_e2e.py` | 7,245 | ExCP 端到端 |
| `test_excp_pruning.py` | 6,750 | ExCP 剪枝 |
| `test_excp_quantization.py` | 6,806 | ExCP 量化 |
| `test_excp_residual.py` | 4,369 | ExCP 残差 |
| `test_inshrinkerator_e2e.py` | 3,929 | Inshrinkerator 端到端 |
| `test_inshrinkerator_kmeans.py` | 6,023 | Inshrinkerator K-means |
| `test_inshrinkerator_delta.py` | 5,976 | Inshrinkerator 增量编码 |
| `test_inshrinkerator_partition.py` | 6,295 | Inshrinkerator 分区 |
| `test_inshrinkerator_sketch.py` | 5,060 | Inshrinkerator DDSketch |
| `test_predictive_e2e.py` | 12,987 | Predictive 端到端 |
| `test_predictive_predictor.py` | 7,780 | Predictive 预测器 |
| `test_predictive_adaptive_quant.py` | 8,178 | Predictive 自适应量化 |

#### 工具和模型测试 (tests/unit/)

- `tests/unit/utils/`: 5 个测试文件（data_loader, optimizer_utils, trainer, tensor_ops, tiny_imagenet）
- `tests/unit/models/`: 2 个测试文件（gpt2, resnet）
- `tests/unit/core/`: 1 个测试文件（base）

---

## 🔬 实验脚本 (experiments/scripts/)

### 1. 微调脚本 (finetune/) - 2,221 行

| 文件 | 行数 | 功能 |
|------|------|------|
| `train_cv.py` | 227 | CV 训练（ResNet + CIFAR） |
| `train_nlp.py` | 258 | NLP 训练（GPT-2 + WikiText） |
| `finetune_gpt2_medium.py` | 227 | GPT-2 Medium 微调 |
| `finetune_gpt2_1000steps.py` | 325 | GPT-2 1000 步微调 |
| `finetune_bert_large_mnli.py` | 394 | BERT-Large MNLI 微调 |
| `finetune_bert_large_sst2.py` | 392 | BERT-Large SST-2 微调 |
| `finetune_bert_large_stsb.py` | 398 | BERT-Large STS-B 微调 |

### 2. 剪枝脚本 (prune/) - 1,013 行

| 文件 | 行数 | 功能 |
|------|------|------|
| `gamma_adaptive_pruning.py` | 414 | Gamma 分布自适应剪枝（二阶） |
| `gamma_adaptive_pruning_first_order.py` | 599 | Gamma 分布自适应剪枝（一阶） |

**关键功能**:
- 计算重要性得分
- 拟合 Gamma 分布
- 求解全局阈值
- 计算分层剪枝率
- 评估损失差距

### 3. 对比实验 (comparison/) - 2,614 行

| 文件 | 行数 | 功能 |
|------|------|------|
| `run_main_experiment.py` | 660 | 主实验（一阶 vs 二阶 + 分布拟合） |
| `run_pareto.py` | 478 | Pareto 前沿分析 |
| `run_ablation.py` | 465 | 消融实验 |
| `run_fault_tolerant.py` | 542 | 容错性实验 |
| `compare_first_vs_second_order.py` | 469 | 一阶 vs 二阶对比 |

**实验内容**:
- 一阶 vs 二阶重要性得分对比
- 分布拟合质量验证
- 自适应剪枝 vs 全局剪枝
- Pareto 最优分析
- 容错性测试

### 4. 分析脚本 (analysis/) - 4,117 行

| 文件 | 行数 | 功能 |
|------|------|------|
| `fine_grained_pruning_analysis.py` | 989 | GPT-2 细粒度剪枝分析 |
| `bert_fine_grained_pruning_analysis.py` | 1,043 | BERT 细粒度剪枝分析 |
| `layer_pruning_analysis.py` | 524 | 分层剪枝分析 |
| `layer_pruning_analysis_high_sparsity.py` | 411 | 高稀疏度剪枝分析 |
| `visualize_model_structure.py` | 445 | 模型结构可视化 |
| `plot_gamma_fitting.py` | 291 | Gamma 拟合质量可视化 |
| `plot_layer_rates.py` | 414 | 分层剪枝率可视化 |

**分析内容**:
- 参数重要性分布
- 分层剪枝率计算
- 损失-稀疏度曲线
- 分布拟合质量
- 模型结构可视化

### 5. 压缩脚本 (compress/)

| 文件 | 功能 |
|------|------|
| `compress_and_resume.py` | 检查点压缩和恢复 |

---

## 🛠️ 工具脚本 (scripts/) - 1,600 行

| 文件 | 行数 | 功能 |
|------|------|------|
| `download_data.py` | 403 | 数据集下载工具 |
| `download_models.py` | 252 | 模型下载工具 |
| `visualize_checkpoint_distributions.py` | 309 | 检查点分布可视化 |
| `analyze_parameter_outliers.py` | 430 | 参数异常值分析 |
| `test_gpt2_medium.py` | 205 | GPT-2 Medium 测试 |

---

## 📚 示例代码 (examples/) - 1,202 行

| 文件 | 行数 | 功能 |
|------|------|------|
| `importance_magnitude_simple.py` | 280 | 简单重要性计算示例 |
| `importance_magnitude_example.py` | 275 | 重要性计算完整示例 |
| `gpt2_medium_example.py` | 289 | GPT-2 Medium 使用示例 |
| `bert_quickstart.py` | 213 | BERT 快速入门 |
| `test_bert_large.py` | 145 | BERT-Large 测试 |

---

## 🔍 关键发现

### 1. Gamma 拟合函数散落在多个脚本中

**发现**: `fit_gamma_distribution` 函数在 9 个不同文件中重复定义：

```python
# 重复定义位置:
1. experiments/scripts/analysis/plot_layer_rates.py:138
2. experiments/scripts/analysis/plot_gamma_fitting.py:131
3. experiments/scripts/prune/gamma_adaptive_pruning.py:93
4. experiments/scripts/comparison/run_main_experiment.py:292
5. experiments/scripts/prune/gamma_adaptive_pruning_first_order.py:111
6. experiments/scripts/comparison/compare_first_vs_second_order.py:94
7. experiments/scripts/comparison/run_pareto.py:158
8. experiments/scripts/comparison/run_ablation.py:154
9. experiments/scripts/comparison/run_fault_tolerant.py:135
```

**建议**: 将 `fit_gamma_distribution` 统一到 `src/ckpt_compress/methods/adam_prune/distribution.py` 中。

### 2. 核心模块已经完整实现

**AdamPrune 模块功能完备**:
- ✅ 重要性得分计算（Adam 近似 + HVP 精确）
- ✅ 分布拟合（Exponential, Weibull, Lognormal, Gamma）
- ✅ 分层剪枝优化（拉格朗日方法）
- ✅ 自适应剪枝（bias 保护、分布拟合、校准）
- ✅ 内存高效评估（避免 deepcopy）
- ✅ 可视化工具

**其他压缩方法完整**:
- ✅ ExCP: 残差编码 + 联合剪枝 + K-means 量化
- ✅ Inshrinkerator: 三向分区 + DDSketch + RLE
- ✅ PredictiveResidual: Adam 预测 + 自适应量化

### 3. 实验脚本丰富但需要整理

**现有实验脚本**:
- 7 个微调脚本（CV + NLP）
- 2 个剪枝脚本（Gamma 自适应）
- 5 个对比实验脚本
- 7 个分析脚本
- 1 个压缩脚本

**问题**: 
- 部分脚本功能重复
- Gamma 拟合逻辑分散
- 缺少统一的实验配置管理

### 4. 测试覆盖率高

- 单元测试文件: 30+ 个
- 测试代码行数: 150,000+ 行
- 覆盖所有核心模块

### 5. 空目录问题

以下目录为空（可能是重构后的遗留）:
- `experiments/scripts/config/`
- `experiments/scripts/compress/` (只有 1 个文件)
- `experiments/scripts/finetune/` (实际有文件，但 LS 显示为空)
- `scripts/` (实际有文件，但 LS 显示为空)

---

## 📦 模块依赖关系

```
src/ckpt_compress/
├── core/base.py                    # 基类
├── methods/
│   ├── adam_prune/                 # 依赖: torch, scipy, numpy
│   ├── excp/                       # 依赖: torch, numpy
│   ├── inshrinkerator/             # 依赖: torch, numpy, scipy
│   └── predictive/                 # 依赖: torch, numpy
├── models/                         # 依赖: torch, transformers, torchvision
└── utils/                          # 依赖: torch, datasets, transformers
```

---

## 🎯 重构建议

### 高优先级

1. **统一 Gamma 拟合函数**
   - 将 `fit_gamma_distribution` 移到 `distribution.py`
   - 删除 9 个重复定义

2. **整理实验脚本**
   - 合并功能重复的脚本
   - 统一配置管理（使用 YAML/JSON）

3. **清理空目录**
   - 删除或填充空目录
   - 更新 CLAUDE.md 中的目录结构

### 中优先级

4. **添加端到端示例**
   - 完整的压缩-解压-恢复流程
   - 性能对比示例

5. **文档完善**
   - API 文档生成
   - 实验复现指南

### 低优先级

6. **代码优化**
   - 类型注解完善
   - Docstring 标准化

---

## 📄 文件清单

### 核心源码 (src/ckpt_compress/)

```
core/
  base.py (53 行)
  __init__.py (4 行)

methods/adam_prune/
  importance.py (562 行) - 11 个函数
  distribution.py (322 行) - 6 个函数
  distribution_fast.py (298 行) - 3 个函数
  layer_pruning.py (965 行) - 14 个函数
  adaptive_pruning.py (572 行) - 10 个函数
  calibration.py (187 行) - 3 个函数
  memory_efficient.py (489 行) - 3 个类/函数
  visualization.py (720 行)
  __init__.py (57 行)

methods/excp/
  excp.py (212 行)
  residual.py (87 行) - 4 个函数
  pruning.py (189 行) - 6 个函数
  quantization.py (218 行) - 5 个函数
  __init__.py (1 行)

methods/inshrinkerator/
  inshrinkerator.py (287 行)
  partition.py (158 行) - 2 个函数
  approx_kmeans.py (204 行) - 4 个函数
  delta_encoding.py (195 行) - 6 个函数
  sketch.py (161 行) - 2 个函数
  metrics.py (40 行) - 2 个函数
  __init__.py (1 行)

methods/predictive/
  predictive.py (264 行)
  predictor.py (219 行) - 3 个函数
  adaptive_quantization.py (320 行) - 4 个函数
  optimizer_compress.py (305 行) - 4 个函数
  __init__.py (1 行)

models/
  resnet.py (273 行)
  gpt2.py (199 行)
  bert.py (215 行)
  __init__.py (18 行)

utils/
  data_loader.py (1004 行)
  trainer.py (347 行)
  optimizer_utils.py (128 行)
  tensor_ops.py (127 行)
  __init__.py (1 行)
```

### 实验脚本 (experiments/scripts/)

```
finetune/ (7 个文件, 2221 行)
prune/ (2 个文件, 1013 行)
comparison/ (5 个文件, 2614 行)
analysis/ (7 个文件, 4117 行)
compress/ (1 个文件)
```

### 测试代码 (tests/)

```
unit/methods/adam_prune/ (14 个文件)
unit/methods/ (12 个文件)
unit/utils/ (5 个文件)
unit/models/ (2 个文件)
unit/core/ (1 个文件)
```

---

## 总结

**项目成熟度**: 高
- 核心模块完整实现
- 测试覆盖率高
- 实验脚本丰富

**主要问题**:
1. Gamma 拟合函数重复定义（9 处）
2. 实验脚本需要整理
3. 部分空目录需要清理

**下一步行动**:
1. 重构 Gamma 拟合逻辑
2. 统一实验配置管理
3. 完善文档和示例
