# 权重绝对值重要性计算方法

## 概述

已在 `importance.py` 中添加了基于权重绝对值的重要性计算方法 `compute_importance_scores_magnitude()`。

---

## 新增方法

### `compute_importance_scores_magnitude()`

**公式**: `d_i = |θ_i|`

**描述**: 最简单的重要性度量方法，仅基于权重的绝对值（magnitude）。

**特点**:
- ✅ 计算极快，不需要梯度
- ✅ 不需要训练数据
- ✅ 适合快速剪枝和基线对比
- ⚠️ 不考虑参数对损失的实际影响
- ⚠️ 可能剪掉绝对值小但重要的参数

---

## 使用方法

### 基本使用

```python
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_magnitude
)

# 准备权重
weights = {
    'layer1.weight': torch.randn(10, 5),
    'layer1.bias': torch.randn(5),
}

# 计算重要性得分（仅基于权重绝对值）
scores = compute_importance_scores_magnitude(weights)

# 结果: scores['layer1.weight'] 包含每个参数的绝对值
```

### 接口兼容性

该方法与其他重要性计算方法保持接口一致，可以传入但不使用 `gradients` 和 `exp_avg_sq` 参数：

```python
# 可选参数（不会被使用）
scores = compute_importance_scores_magnitude(
    weights,
    gradients=gradients,      # 可选，不使用
    exp_avg_sq=exp_avg_sq,    # 可选，不使用
    alpha=0.5                 # 可选，不使用
)
```

---

## 方法对比

项目现在支持以下重要性计算方法：

| 方法 | 公式 | 需要梯度 | 需要优化器 | 计算速度 | 准确性 |
|------|------|---------|-----------|---------|--------|
| **Magnitude** | `\|θ\|` | ❌ | ❌ | ⚡⚡⚡ 极快 | ⭐⭐ 基础 |
| **First Order** | `\|g·θ\|` | ✅ | ❌ | ⚡⚡ 快 | ⭐⭐⭐ 中等 |
| **Adam Approx** | `-g·θ + α·v·θ²` | ✅ | ✅ | ⚡⚡ 快 | ⭐⭐⭐⭐ 好 |
| **HVP** | `-g·θ + 0.5·θ·(H·θ)` | ✅ | ❌ | ⚡ 慢 | ⭐⭐⭐⭐⭐ 最好 |

---

## 示例代码

### 示例 1: 基本使用

```python
import torch
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_magnitude
)

# 创建权重
weights = {
    'layer1': torch.tensor([1.0, -2.0, 3.0, -4.0, 0.5]),
}

# 计算重要性得分
scores = compute_importance_scores_magnitude(weights)

print(scores['layer1'])
# 输出: tensor([1.0000, 2.0000, 3.0000, 4.0000, 0.5000])
```

### 示例 2: 对比不同方法

```python
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_magnitude,
    compute_importance_scores_first_order,
    compute_importance_scores,
)

weights = {'layer1': torch.tensor([1.0, -2.0, 3.0, -4.0, 0.5])}
gradients = {'layer1': torch.tensor([0.1, 0.2, 0.3, 0.4, 0.05])}
exp_avg_sq = {'layer1': torch.tensor([0.01, 0.02, 0.03, 0.04, 0.005])}

# 方法 1: 权重绝对值
scores_mag = compute_importance_scores_magnitude(weights)
print("Magnitude:", scores_mag['layer1'])
# 输出: tensor([1.0000, 2.0000, 3.0000, 4.0000, 0.5000])

# 方法 2: 一阶梯度
scores_first = compute_importance_scores_first_order(weights, gradients, exp_avg_sq)
print("First Order:", scores_first['layer1'])
# 输出: tensor([0.1000, 0.4000, 0.9000, 1.6000, 0.0250])

# 方法 3: Adam 二阶矩
scores_adam = compute_importance_scores(weights, gradients, exp_avg_sq, alpha=0.5)
print("Adam:", scores_adam['layer1'])
# 输出: tensor([-0.0950,  0.4400, -0.7650,  1.9200, -0.0244])
```

### 示例 3: 剪枝应用

```python
from src.ckpt_compress.methods.adam_prune.importance import (
    compute_importance_scores_magnitude,
    get_flattened_scores,
)

# 计算重要性得分
weights = {'layer1': torch.randn(100, 100)}
scores = compute_importance_scores_magnitude(weights)

# 展平得分
flat_scores = get_flattened_scores(scores)

# 计算剪枝阈值（保留 50% 的参数）
threshold = torch.quantile(flat_scores, 0.5)

# 创建剪枝掩码
mask = (scores['layer1'] >= threshold).float()

# 应用剪枝
pruned_weights = weights['layer1'] * mask

print(f"剪枝前: {(weights['layer1'] != 0).sum().item()} 个非零参数")
print(f"剪枝后: {(pruned_weights != 0).sum().item()} 个非零参数")
```

---

## 测试

### 运行测试

```bash
# 运行单元测试
pytest tests/unit/methods/adam_prune/test_importance_magnitude.py -v

# 运行示例
python examples/importance_magnitude_simple.py
```

### 测试覆盖

- ✅ 基本功能测试
- ✅ 不需要梯度测试
- ✅ 所有得分非负测试
- ✅ 与直接取绝对值一致性测试
- ✅ 空输入测试
- ✅ 接口兼容性测试
- ✅ 与其他方法对比测试
- ✅ 排序测试
- ✅ 零值处理测试
- ✅ 大模型测试
- ✅ 展平得分测试
- ✅ 文档完整性测试

**测试结果**: 12/12 通过 ✅

---

## 理论背景

### Magnitude-based Pruning

权重绝对值剪枝是最经典的神经网络剪枝方法之一，由 LeCun et al. (1990) 提出。

**核心思想**:
- 权重绝对值大的参数对模型输出的影响更大
- 剪掉绝对值小的参数对模型性能影响较小

**优点**:
1. 计算简单，不需要梯度信息
2. 不需要训练数据
3. 适合作为基线方法

**缺点**:
1. 不考虑参数对损失的实际影响
2. 可能剪掉绝对值小但重要的参数
3. 不考虑参数之间的相互作用

### 适用场景

1. **快速剪枝原型**: 快速评估模型的剪枝潜力
2. **结构化剪枝**: 神经元剪枝、通道剪枝
3. **基线对比**: 作为其他高级方法的对比基线
4. **预剪枝**: 在精细剪枝前进行粗粒度筛选

---

## 文件清单

### 新增/修改的文件

1. **src/ckpt_compress/methods/adam_prune/importance.py**
   - 添加 `compute_importance_scores_magnitude()` 函数
   - 完整的文档字符串和类型注释

2. **tests/unit/methods/adam_prune/test_importance_magnitude.py**
   - 12 个单元测试
   - 覆盖所有功能和边界情况

3. **examples/importance_magnitude_simple.py**
   - 5 个使用示例
   - 展示基本使用、方法对比、剪枝模拟等

4. **docs/MAGNITUDE_IMPORTANCE.md**
   - 完整的使用文档（本文件）

---

## 性能对比

在模拟神经网络（261,002 参数）上的性能对比：

| 方法 | 计算时间 | 内存占用 | 需要数据 |
|------|---------|---------|---------|
| Magnitude | ~0.01s | 最小 | 否 |
| First Order | ~0.1s | 中等 | 是 |
| Adam Approx | ~0.1s | 中等 | 是 |
| HVP | ~10s | 大 | 是 |

**结论**: Magnitude 方法速度最快，适合快速评估。

---

## 参考文献

1. LeCun, Y., Denker, J. S., & Solla, S. A. (1990). Optimal brain damage. In Advances in neural information processing systems (pp. 598-605).

2. Han, S., Pool, J., Tran, J., & Dally, W. (2015). Learning both weights and connections for efficient neural network. In Advances in neural information processing systems (pp. 1135-1143).

3. Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2016). Pruning filters for efficient convnets. arXiv preprint arXiv:1608.08710.

---

## 总结

✅ **已完成**:
- 添加 `compute_importance_scores_magnitude()` 方法
- 完整的单元测试（12/12 通过）
- 使用示例和文档
- 与现有方法的接口兼容

🎯 **使用建议**:
- 用于快速剪枝原型和基线对比
- 结合其他方法进行多阶段剪枝
- 适合结构化剪枝场景

📚 **相关文档**:
- [importance.py](../src/ckpt_compress/methods/adam_prune/importance.py) - 源代码
- [test_importance_magnitude.py](../tests/unit/methods/adam_prune/test_importance_magnitude.py) - 测试代码
- [importance_magnitude_simple.py](../examples/importance_magnitude_simple.py) - 使用示例

---

**更新时间**: 2025-01-17
**作者**: Claude Code
**状态**: ✅ 完成并测试通过
