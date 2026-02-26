# AdamPrune: 基于Adam二阶矩的理论剪枝方法

## 方法概述

AdamPrune 是一种基于理论推导的剪枝方法，利用 Adam 优化器的二阶矩作为 Hessian 对角线的代理，
通过数学公式直接计算给定损失容忍度下的最优剪枝比例。

## 核心理论

### 1. 重要性得分 (Importance Score)

对于每个参数 θ_i，其重要性得分定义为：

```
s_i = |g_i * θ_i| + 0.5 * v_i * θ_i²
```

其中：
- g_i: 参数的梯度
- θ_i: 参数值
- v_i: Adam 的二阶矩 (exp_avg_sq)

**理论基础**：
- Adam 的二阶矩 v_i 是 Hessian 对角线 H_ii 的无偏估计
- 通过 Fisher 信息矩阵与 Hessian 的联系：E[v_i] ≈ H_ii
- 重要性得分是剪枝该参数导致的损失变化的上界

### 2. 最优剪枝比例公式

给定可容忍的损失增量 ε，最优剪枝比例为：

```
p* = sqrt(2ε / (N * s̄))
```

其中：
- N: 参数总数
- s̄: 平均重要性得分 = (1/N) * Σs_i
- ε: 可容忍的损失增量

**推导假设**：
- 假设 4（指数分布）：重要性得分服从指数分布 s_i ~ Exp(λ)
- 小 p 近似：当 p << 1 时，累积损失 ΔL(p) ≈ Np²/(2λ)

### 3. 理论验证目标

本实验旨在验证：
1. **假设 4 的有效性**：实际重要性得分分布是否接近指数分布？
2. **公式的准确性**：给定 ε，计算出的 p* 剪枝后，实际损失增量是否接近 ε？

## 模块设计

```
adam_prune/
├── __init__.py
├── README.md                 # 本文档
├── importance.py             # 重要性得分计算
├── pruning.py               # 剪枝比例计算和执行
├── distribution.py          # 分布分析和拟合
└── adam_prune.py            # 主压缩器（整合所有模块）
```

## 核心接口

### 1. importance.py

```python
def compute_importance_scores(
    weights: Dict[str, torch.Tensor],
    gradients: Dict[str, torch.Tensor],
    exp_avg_sq: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    计算每个参数的重要性得分。

    s_i = |g_i * θ_i| + 0.5 * v_i * θ_i²
    """
    pass
```

### 2. pruning.py

```python
def compute_optimal_prune_ratio(
    importance_scores: Dict[str, torch.Tensor],
    epsilon: float
) -> float:
    """
    根据公式计算最优剪枝比例。

    p* = sqrt(2ε / (N * s̄))
    """
    pass

def prune_by_importance(
    weights: Dict[str, torch.Tensor],
    importance_scores: Dict[str, torch.Tensor],
    prune_ratio: float
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    按重要性得分剪枝，返回剪枝后的权重和剪枝掩码。
    """
    pass
```

### 3. distribution.py

```python
def fit_exponential_distribution(
    importance_scores: torch.Tensor
) -> Tuple[float, float]:
    """
    拟合指数分布，返回 (lambda, goodness_of_fit)。
    """
    pass

def analyze_distribution(
    importance_scores: torch.Tensor
) -> Dict[str, Any]:
    """
    分析重要性得分的分布特性。
    返回：均值、方差、偏度、峰度、与指数分布的拟合度等。
    """
    pass
```

## 实验设计

### 实验 1：分布验证

**目标**：验证假设 4（指数分布假设）

**步骤**：
1. 在 CIFAR-10 上训练 ResNet18 若干轮
2. 提取权重、梯度、Adam 二阶矩
3. 计算所有参数的重要性得分
4. 分析分布：绘制直方图、拟合指数分布、计算 KS 检验统计量

**预期结果**：
- 如果分布接近指数分布，KS 检验 p-value > 0.05
- 如果不是指数分布，分析实际分布类型

### 实验 2：公式验证

**目标**：验证剪枝比例公式的准确性

**步骤**：
1. 设定多个 ε 值（如 0.001, 0.01, 0.1）
2. 对每个 ε，使用公式计算 p*
3. 按 p* 剪枝模型
4. 测量实际损失增量 ΔL
5. 比较 ΔL 与 ε

**预期结果**：
- 如果公式准确，ΔL ≈ ε（误差在合理范围内）
- 记录 ΔL/ε 的比值，分析公式的保守/激进程度

### 实验 3：训练恢复验证

**目标**：验证剪枝后模型能否正常恢复训练

**步骤**：
1. 使用公式计算的 p* 剪枝模型
2. 从剪枝后的检查点恢复训练
3. 比较与原始模型的训练曲线

## 测试用例设计（TDD）

### 单元测试

1. **importance.py 测试**
   - test_importance_score_formula_correct: 验证公式计算正确
   - test_importance_score_all_zeros: 全零输入处理
   - test_importance_score_shape_preserved: 输出形状正确
   - test_importance_score_non_negative: 得分非负

2. **pruning.py 测试**
   - test_prune_ratio_formula_correct: 验证公式计算正确
   - test_prune_ratio_bounds: 剪枝比例在 [0, 1] 范围内
   - test_prune_by_importance_correct_count: 剪枝数量正确
   - test_prune_lowest_importance: 剪枝的是最低重要性的参数

3. **distribution.py 测试**
   - test_fit_exponential_known_data: 对已知指数分布数据拟合
   - test_analyze_distribution_statistics: 统计量计算正确

### 集成测试

1. **端到端测试**
   - test_adam_prune_roundtrip: 完整压缩/解压流程
   - test_adam_prune_loss_bounded: 损失增量在预期范围内

## 依赖

- PyTorch >= 1.9
- NumPy
- SciPy (用于分布拟合和统计检验)

## 参考文献

- 论文：基于 Adam 二阶矩的理论剪枝（NeurIPS 2024 投稿）
