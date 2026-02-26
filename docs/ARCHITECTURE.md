# 架构设计文档

本文档详细说明 `ckpt-compress` 项目的架构设计、各模块实现细节和核心概念。

## 目录

- [核心抽象](#核心抽象)
- [压缩方法详解](#压缩方法详解)
- [工具模块](#工具模块)
- [核心概念](#核心概念)

## 核心抽象

### BaseCompressor

所有压缩方法都继承自 `src/ckpt_compress/core/base.py` 中的 `BaseCompressor` 抽象类：

```python
class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, state_dict: Dict[str, torch.Tensor]) -> bytes:
        """压缩模型状态字典"""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> Dict[str, torch.Tensor]:
        """解压字节数据恢复状态字典"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """返回方法名称"""
        pass
```

**设计原则**:
- 统一接口：所有方法使用相同的输入输出格式
- 状态字典：接受 PyTorch 标准的 `Dict[str, torch.Tensor]` 格式
- 字节输出：压缩结果为字节流，便于存储和传输

## 压缩方法详解

### 1. ExCP

**位置**: `methods/excp/excp.py`

**核心理念**: 残差编码 + 权重-动量联合剪枝 + K-means 量化

**主要类**:
- `ExCPConfig`: 配置类
  - `alpha`: 权重剪枝阈值系数
  - `beta`: 动量剪枝阈值系数
  - `p`: 剪枝百分比
  - `n_bits`: 量化位数（默认 4）
  - `use_gzip`: 是否使用 gzip 二次压缩
- `ExCPCompressor`: 主压缩器类

**子模块**:

1. **`pruning.py`**: 联合剪枝
   - `joint_prune(dW, v, m, alpha, beta, p)`: 权重和动量的联合剪枝
   - 剪枝条件：`|dW_i| < alpha * threshold_w OR |v_i| < beta * threshold_v`

2. **`quantization.py`**: K-means 量化
   - `kmeans_quantize_nonzero()`: 对非零值进行 K-means 聚类量化
   - `pack_int4()` / `unpack_int4()`: 4-bit 索引打包/解包

3. **`residual.py`**: 残差计算（当前未使用）
   - `compute_residual()`: 计算权重残差
   - `reconstruct_from_residual()`: 从残差重建权重

**压缩流程**:
```
1. 计算残差: dW = W_t - W_{t-1}
   - 首个检查点: 直接存储完整权重
   - 后续检查点: 仅存储残差

2. 联合剪枝: joint_prune(dW, v_t, m_t, alpha, beta, p)
   - 根据权重和动量的重要性联合决定剪枝

3. K-means 量化: 对非零值聚类
   - 聚类数: 2^n_bits (默认 16)
   - 存储: 聚类中心 + 索引

4. 打包: 将索引打包为 4-bit
   - 两个索引共享一个字节

5. 可选 gzip 压缩
```

**检查点链**:
- ExCP 使用检查点链技术，需要维护前一检查点状态
- 首个检查点存储完整权重，后续仅存储残差

### 2. Inshrinkerator

**位置**: `methods/inshrinkerator/inshrinkerator.py`

**核心理念**: 三向分区 + DDSketch 近似 K-means + RLE 增量编码

**主要类**:
- `InshrinkeratorConfig`: 配置类
  - `n_bins`: 量化桶数
  - `protect_fraction`: 保护参数比例
  - `prune_fraction`: 剪枝参数比例
  - `use_gzip`: 是否使用 gzip
- `InshrinkeratorCompressor`: 主压缩器类

**子模块**:

1. **`partition.py`**: 三向分区
   - `partition()`: 将参数分为三类
     - **保护区**: 高重要性参数，保持全精度
     - **剪枝区**: 低重要性参数，直接置零
     - **量化区**: 中等重要性参数，量化存储
   - `PartitionConfig` / `PartitionResult`: 分区配置和结果

2. **`sketch.py`**: DDSketch 分位数估计
   - `QuantileSketch`: 分位数草图类
     - 使用对数桶映射：`bucket_id = floor(log(x) / log(gamma))`
     - Gamma 参数：`gamma = (1 + alpha) / (1 - alpha)`
     - 提供有界相对误差的分位数查询（默认 1% 误差）
   - `compute_gamma()`: 计算 gamma 参数
   - `compute_bucket_index()`: 计算桶索引

3. **`approx_kmeans.py`**: 基于草图的近似 K-means
   - `approx_kmeans()`: 使用草图直方图进行快速 K-means
   - `weighted_kmeans_plusplus_init()`: 加权 K-means++ 初始化
   - `compute_sample_weights()`: 结合频率和幅度的样本权重
   - `quantize_to_centers()`: 将值量化到最近的中心

4. **`delta_encoding.py`**: 增量编码
   - `delta_encode()` / `delta_decode()`: 增量编码/解码
   - `rle_encode()` / `rle_decode()`: 游程编码/解码

5. **`metrics.py`**: 重要性度量函数

**压缩流程**:
```
1. 参数分区:
   - 计算重要性度量
   - 使用 QuantileSketch 估计分位数阈值
   - 分为保护/剪枝/量化三区

2. 量化区处理:
   - 使用 approx_kmeans() 计算聚类中心
   - 量化到最近的中心

3. 增量编码（如果有前一检查点）:
   - delta_encode(): 计算与前一检查点的差值
   - rle_encode(): 对差值进行游程编码

4. 序列化并可选 gzip 压缩
```

### 3. PredictiveResidual

**位置**: `methods/predictive/predictive.py`

**核心理念**: Adam 权重预测 + 敏感度自适应量化 + 优化器状态压缩

**主要类**:
- `PredictiveConfig`: 配置类
  - `lr`: 学习率
  - `eps`: Adam epsilon
  - `min_bits` / `max_bits`: 量化位数范围
  - `optimizer_bits`: 优化器状态量化位数
  - `use_gzip`: 是否使用 gzip
- `PredictiveCompressor`: 主压缩器类

**子模块**:

1. **`predictor.py`**: Adam 权重预测
   - `AdamPredictor`: 使用 Adam 更新规则预测下一个权重
     - `predict()`: 预测 W_pred = W_t - lr * m_t / (sqrt(v_t) + eps)
     - `compute_residual()`: 计算预测残差 residual = W_actual - W_pred
     - `reconstruct()`: 从残差重建权重

2. **`adaptive_quantization.py`**: 敏感度自适应量化
   - `AdaptiveQuantizer`: 基于敏感度的自适应量化
     - 敏感度估计：使用梯度幅度作为代理
     - 位数分配：高敏感度参数分配更多位数
     - `quantize()`: 根据敏感度分配位数并量化
     - `dequantize()`: 反量化

3. **`optimizer_compress.py`**: 优化器状态压缩
   - `OptimizerStateCompressor`: 压缩 exp_avg 和 exp_avg_sq
     - exp_avg: 使用对称量化（可正可负）
     - exp_avg_sq: 使用对数空间量化（始终为正）
     - `compress()` / `decompress()`: 压缩/解压

**压缩流程**:
```
1. 权重预测:
   - 使用 AdamPredictor 预测 W_pred
   - 计算预测残差 residual = W_actual - W_pred

2. 敏感度自适应量化:
   - 估计每个参数的敏感度
   - 根据敏感度分配量化位数（min_bits 到 max_bits）
   - 对残差进行量化

3. 优化器状态压缩:
   - exp_avg: 对称量化
   - exp_avg_sq: 对数空间量化

4. 序列化并可选 gzip 压缩
```

### 4. AdamPrune

**位置**: `methods/adam_prune/`

**核心理念**: 利用 Adam 二阶矩作为 Hessian 对角线代理的理论剪枝方法

**模块架构**:

#### `importance.py`: 重要性得分计算

支持两种计算方法：

**1. Adam 二阶矩近似（默认）**:
```python
s_i = -g_i * θ_i + α * v_i * θ_i²
```
- 使用 Adam 的 `exp_avg_sq` 作为 Hessian 对角线近似
- 计算快速，但只考虑对角元素
- 函数：`compute_importance_scores()`

**2. HVP（Hessian-Vector Product）精确计算**:
```python
s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i
```
- 使用真实的 Hessian 矩阵
- 考虑参数间相互作用（非对角元素）
- 需要两次反向传播
- 函数：`compute_importance_scores_hvp()` 和 `compute_importance_scores_hvp_memory_efficient()`

**理论依据**:
删除参数 θ_i（设为0）对损失的泰勒展开：
```
ΔL ≈ -g^T·θ + 0.5·θ^T·H·θ
```

一阶项 `-g_i·θ_i` 保留符号：
- 若 > 0：删除该参数会增加损失（参数重要）
- 若 < 0：删除该参数会减少损失（参数可能有害）

#### `distribution.py` / `distribution_fast.py`: 分布拟合

- `fit_multiple_distributions()`: 拟合多种分布
  - 韦伯分布（Weibull）：最常用
  - 指数分布（Exponential）
  - 对数正态分布（Lognormal）
  - 伽马分布（Gamma）
- `get_best_fit()`: 返回最佳拟合分布（基于 KS 检验）
- `fit_distributions_per_layer_fast()`: GPU 加速的快速拟合
- `fit_weibull_linear_regression()`: 韦伯分布的线性回归拟合

#### `layer_pruning.py`: 分层剪枝优化

- `LayerPruningOptimizer`: 核心优化器类
  - `optimize_with_distributions()`: 基于分布优化的剪枝比例计算
    - 使用拉格朗日乘子法
    - 在损失约束下最大化稀疏度
  - `compute_distribution_loss()`: 计算理论损失
  - `apply_pruning()`: 应用剪枝
- `prune_by_global_sparsity()`: 按全局稀疏度剪枝
- `prune_by_layer_sparsity()`: 按层稀疏度剪枝
- `compute_real_loss_increase()`: 计算真实损失增量

#### `adaptive_pruning.py`: 自适应剪枝辅助

- `filter_prunable_layers()`: 过滤可剪枝层（保留 bias）
- `fit_distributions_per_layer()`: 每层分布拟合
- `build_layer_dist_params()`: 构建层分布参数
- `adjust_epsilon_for_target()`: 调整 epsilon 以达到目标损失

#### `calibration.py`: 幂律校正

当理论预测与实际损失存在系统性偏差时，使用幂律模型进行校正：

```
拟合: y = a * x^b
其中: x = p / L0（被剪枝参数重要性之和 / 基线损失）
     y = Δloss / L0（实际相对损失增量）
```

- `fit_powerlaw()`: 拟合幂律模型
- `invert_powerlaw()`: 幂律逆映射
- `compute_calibration_variables()`: 计算校正变量
- `compute_calibration_metrics()`: 计算拟合质量指标（R², RMSE, Spearman）

#### `visualization.py`: 可视化工具

- `plot_layer_distributions()`: 层重要性分布图
- `plot_marginal_loss_curves()`: 边际损失曲线
- `plot_calibration_scatter()`: 校正散点图（log-log）

#### `memory_efficient.py`: 内存高效的剪枝评估

**关键类**:

1. **`MemoryMonitor`**: 内存监控和限制
   - `get_memory_usage()`: 获取当前内存使用量
   - `check_memory_limit()`: 检查是否超出限制
   - `log_memory_usage()`: 记录内存使用量

2. **`MemoryEfficientEvaluator`**: 使用原地修改代替 deepcopy
   - `save_original_weights()`: 保存原始权重
   - `apply_pruned_weights()`: 应用剪枝后的权重
   - `restore_original_weights()`: 恢复原始权重
   - `temporary_weights()`: 临时应用剪枝权重的上下文管理器
   - `evaluate_loss()`: 评估模型损失
   - `evaluate_loss_increase()`: 评估剪枝后的损失增量

3. **`MemoryEfficientDataCollector`**: 收集模型数据
   - `collect()`: 收集权重、梯度和优化器状态
   - 可选 CPU 卸载以节省 GPU 内存

4. **`run_sparsity_grid_memory_efficient()`**: 内存优化的稀疏度网格实验
   - 使用原地修改 + 恢复策略
   - 强制垃圾回收
   - 内存监控和限制

**为什么需要内存高效评估**:
- GPT-2 模型参数量大（~124M）
- 使用 `deepcopy` 会导致内存泄漏和 OOM
- 原地修改 + 恢复策略大幅减少内存使用

## 工具模块

### `utils/tensor_ops.py`

张量操作工具：

- `TensorSpec`: 张量规格说明类
  - 记录张量的形状、数据类型、设备
- `FlattenSpec`: 展平规格说明类
  - 记录如何将状态字典展平为一维向量
- `flatten_state_dict()`: 将状态字典展平为一维向量
- `unflatten_state_dict()`: 从一维向量还原状态字典

### `utils/optimizer_utils.py`

优化器状态操作工具。

### `utils/data_loader.py`

数据加载器：

- `get_cifar10_loaders()`: CIFAR-10 数据加载器
  - 支持数据增强
  - 支持子集采样
- `get_wikitext2_dataloader()`: WikiText-2 数据加载器
  - 支持 `local_path` 参数加载本地文件
  - 支持 HuggingFace 缓存配置
- `WikiText2Dataset`: WikiText-2 数据集封装类
  - 处理文本编码
  - 序列切分

### 模型

**`models/resnet.py`**: CIFAR-10 实验用的 ResNet 实现

**`models/gpt2.py`**:
- `GPT2ForExperiment`: GPT-2 模型封装
  - 提供统一的实验接口
  - 封装 HuggingFace 的 GPT2LMHeadModel
- `get_gpt2_small()`: 工厂函数，创建 GPT-2 small 模型

## 核心概念

### 状态字典格式

所有方法都接受 PyTorch 状态字典：`Dict[str, torch.Tensor]`

键为参数名称，例如：
- `"transformer.h.0.attn.c_attn.weight"`
- `"transformer.h.0.attn.c_attn.bias"`
- `"transformer.h.0.mlp.c_fc.weight"`

### 检查点链（Checkpoint Chaining）

ExCP 和 PredictiveResidual 使用检查点链技术：

- **首个检查点**: 存储完整权重
- **后续检查点**: 仅存储残差/差值

**优势**:
- 利用连续检查点之间的相关性
- 大幅减少存储空间

**限制**:
- 需要维护前一检查点的状态
- 解压时需要按顺序解压

### 层类型提取

AdamPrune 使用 `extract_layer_type()` 函数对参数进行分类：

- `"c_attn"`, `"c_proj"`: Attention 层
- `"mlp"`: FFN 层
- `"ln_"`: Layer Norm 层

用于按层类型进行聚合分析。

### Bias 处理

AdamPrune 默认保留 bias 参数（不剪枝）：

- `filter_prunable_layers()`: 过滤掉 bias 参数
- `is_bias_param()`: 判断是否为 bias 参数

**原因**:
- Bias 参数量少，剪枝收益小
- Bias 对模型性能影响较大

### DDSketch 算法

Inshrinkerator 使用 DDSketch（分布式数据草图）算法进行高效分位数估计。

**核心思想**:
- 使用对数桶映射将值分配到桶中：`bucket_id = floor(log(x) / log(gamma))`
- Gamma 参数控制精度：`gamma = (1 + alpha) / (1 - alpha)`
- 提供有界相对误差的分位数查询（默认 alpha=0.01，即 1% 误差）

**优势**:
- 空间复杂度低，适合大规模数据
- 支持增量更新和草图合并
- 用于快速计算阈值，避免全局排序

**应用**:
- 计算分区阈值（保护/剪枝/量化）
- 快速估计分位数

### 内存管理策略

对于大模型（如 GPT-2）实验，内存管理至关重要。

#### 批次缓存

- 使用 `cache_batches()` 确保梯度计算和损失评估使用相同数据
- 避免重复加载数据导致的内存峰值

#### 内存高效评估（`memory_efficient.py`）

**核心策略**:
1. **原地修改 + 恢复**: 代替 `deepcopy`
   ```python
   # 错误做法（会导致内存泄漏）
   model_copy = copy.deepcopy(model)

   # 正确做法（使用 MemoryEfficientEvaluator）
   evaluator.save_original_weights()
   evaluator.apply_pruned_weights(pruned_weights)
   loss = evaluate(model)
   evaluator.restore_original_weights()
   ```

2. **内存监控**: `MemoryMonitor` 监控内存使用，防止 OOM

3. **CPU 卸载**: 可选将数据移动到 CPU 以节省 GPU 显存

4. **强制垃圾回收**: 及时释放内存

#### 参数控制

- `--max_samples`: 限制数据集大小
- `--num_batches`: 缓存的批次数
- `--batch_size`: 每批次大小
- `--seq_length`: 序列长度（影响显存占用）
- `--memory_limit_gb`: 内存限制（默认 30GB）

### 分布拟合

AdamPrune 支持拟合多种分布类型：

- **韦伯分布（Weibull）**: 最常用的拟合分布
  - 形状参数 k 和尺度参数 λ
  - 适用于大多数层的重要性分布
- **指数分布（Exponential）**: 简单但可能不够准确
- **对数正态分布（Lognormal）**: 适用于某些层的分布
- **伽马分布（Gamma）**: 另一个可选分布

拟合质量通过 KS 检验（Kolmogorov-Smirnov test）评估。

### 幂律校正

当理论预测与实际损失存在系统性偏差时，使用幂律模型进行校正。

**拟合模型**:
```
y = a * x^b
```

其中:
- `x = p / L0`：被剪枝参数重要性之和 / 基线损失
- `y = Δloss / L0`：实际相对损失增量

**逆映射**:
给定目标损失 y*，可通过逆映射计算所需的 x*：
```
x* = (y* / a)^(1/b)
```

**应用**:
- 校正理论预测与实际损失的偏差
- 在损失约束下搜索最优稀疏度

## 占位符目录

以下目录为未来扩展预留，当前仅包含 `__init__.py`：

- **`methods/delta/`**: 增量压缩方法（预留）
- **`methods/quantization/`**: 量化压缩方法（预留）
- **`methods/sparse/`**: 稀疏压缩方法（预留）
- **`methods/general/`**: 通用压缩方法如 gzip, lz4, zstd（预留）
