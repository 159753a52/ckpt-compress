# Inshrinkerator 代码修改计划

## 一、问题分析

经过仔细阅读 Inshrinkerator 论文，确认其剪枝策略是 **Uniform Pruning**：
- 搜索空间中 Pruning Fraction 只有一个全局参数 F（候选值 {0, 0.1, 0.2, 0.3, 0.4, 0.5}）
- 所有层（除 Embedding）各自独立按 importance 排序，剪掉自己的 F% 参数
- 不同层类型之间**没有**差异化的比例分配

但当前代码 `inshrinkerator_search.py` 实现的是 **per-type search**：为每种层类型（attn/mlp/...）搜索不同的剪枝比例，这与论文不符。

### 具体问题清单

| 文件 | 问题 | 严重程度 |
|------|------|---------|
| `inshrinkerator_search.py` | 搜索的是 per-type ratios（每种层类型不同比例），论文实际是单一全局 F | **高** |
| `inshrinkerator_search.py` | `_generate_candidates` 生成的是 per-type 组合（如 attn=0.2, mlp=0.4），搜索空间是 `candidates^num_types`，论文只有 `len(candidates)` 个点 | **高** |
| `inshrinkerator_search.py` | `apply_pruning_per_type` 把同类型层参数汇集后全局排序取阈值，但论文是每层独立排序剪 F% | **中** |
| `per_type_allocation.py` | 整个类的设计（不同层类型不同比例 + 比例缩放）基于错误的理解 | **高** |
| `run_inshrinkerator_like_search.py` | 调用了错误的搜索逻辑 | **高** |
| `run_table1.py` | 包含 `per-type-auto` + `per-type` 方法，基于错误的 per-type 搜索 | **中** |
| `experiment_plan.md` | 已修正（前一轮对话） | 已修复 |

## 二、修改方案

### 2.1 `inshrinkerator_search.py` —— 重写搜索逻辑

**目标**：搜索单一全局剪枝比例 F，与量化/保护参数联合搜索（目前先只做剪枝部分）。

**修改内容**：

1. **简化搜索空间**：候选值从 per-type 组合 改为 单一 F 列表
   ```python
   # 当前（错误）：每种层类型独立搜索
   candidates = product([0.0, 0.2, 0.4], repeat=num_types)  # 3^3 = 27 种
   
   # 修改后（正确）：单一全局 F
   candidates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # 6 种
   ```

2. **简化搜索算法**：
   - 当前的两阶段 coarse + fine grid 搜索过于复杂且前提错误
   - 改为：从最激进的 F 开始，逐个评估，找到满足 ε 的最大 F
   - 可选保留二分搜索加速（利用单调性：F 越大质量越差）

3. **修改 `apply_pruning_per_type`**：
   - 当前：把同类型层参数汇集排序，算一个统一阈值
   - 修改：每层独立排序，各自剪 F%（即直接复用 `pruner.py` 中的 `apply_pruning`）

4. **修改 `SearchResult`**：
   - `per_type_ratios: Dict[str, float]` → `global_prune_ratio: float`
   - 或保留 per_type_ratios 但所有类型值相同

**新的搜索流程**：
```
输入: candidates = [0.5, 0.4, 0.3, 0.2, 0.1, 0.0], epsilon, metric ∈ {magnitude, sensitivity}
对每个 metric:
    对每个 F（从大到小）:
        对每层独立剪 F%
        评估质量损失
        如果 ≤ epsilon:
            记录 (F, metric, CR, quality_drop)
            break（单调性保证后续 F 更小，CR 更低）
返回两个 metric 中 CR 更高的结果
```

### 2.2 `per_type_allocation.py` —— 简化或废弃

**方案**：这个文件的核心功能（从搜索结果JSON加载 per-type ratios 并缩放）基于错误前提。

- **选项A（推荐）**：废弃此文件，Inshrinkerator baseline 直接使用 `UniformAllocation`
- **选项B**：保留但重命名为 `InshrinkeratorAllocation`，内部调用 `UniformAllocation`

推荐选项A，因为 `allocation.py` 中已有 `UniformAllocation`，功能完全一致。

### 2.3 `run_inshrinkerator_like_search.py` —— 更新调用

- 适配新的搜索接口
- 搜索输出改为全局 F 值（而非 per-type ratios JSON）
- 实际上此脚本的必要性降低：既然 Inshrinkerator 就是 Uniform，搜索只是在 6 个 F 值中选最优的，直接在 `run_table1.py` 中遍历即可

### 2.4 `run_table1.py` —— 清理 per-type 方法

- 删除 `per-type-auto` + `per-type` 方法条目
- 删除 `--per_type_json` 参数和相关逻辑
- Inshrinkerator baseline 就是 `first-order + uniform`，已在 METHODS 中

**修改后的 METHODS**：
```python
METHODS = [
    {'importance': 'magnitude',          'allocation': 'uniform'},
    {'importance': 'first-order',        'allocation': 'uniform'},  # Inshrinkerator
    {'importance': 'residual-magnitude', 'allocation': 'uniform'},  # ExCP
    {'importance': 'second-order-hvp',   'allocation': 'uniform'},  # Ours (score only)
    {'importance': 'second-order-hvp',   'allocation': 'gamma-adaptive'},  # Ours (full)
]
```

### 2.5 `pruner.py` 中的 `apply_pruning` —— 确认正确性

当前实现**已经是正确的**：每层独立排序，各自剪 `layer_ratios[name]` 比例。
当 `UniformAllocation` 分配时，所有层 ratio 相同 = 每层各自剪 F%。
无需修改。

### 2.6 `param_schema.py` —— 保留

层类型映射仍然有用：
- `skip` 类型用于排除 embedding/norm/bias（这与论文一致）
- 我们自己的 Gamma-adaptive 方法仍然需要层类型信息

无需修改。

## 三、修改优先级

| 优先级 | 任务 | 影响 |
|--------|------|------|
| P0 | 重写 `inshrinkerator_search.py` 搜索逻辑 | 核心修复 |
| P0 | 更新 `run_table1.py` 删除 per-type 方法 | 实验脚本修复 |
| P1 | 废弃或简化 `per_type_allocation.py` | 清理错误抽象 |
| P1 | 更新 `run_inshrinkerator_like_search.py` | 搜索脚本修复 |
| P2 | 更新 `__init__.py` 导出 | 清理接口 |

## 四、不需要修改的部分

| 文件 | 原因 |
|------|------|
| `allocation.py` (UniformAllocation, GammaAdaptiveAllocation) | 已正确 |
| `importance.py` (所有 scorer) | 已正确 |
| `pruner.py` (apply_pruning, Pruner) | 已正确 |
| `param_schema.py` | 仍需使用，无需改动 |

## 五、验证方法

修改完成后，运行以下验证：
1. `pytest tests/` 确保现有测试通过
2. 快速跑 `run_table1.py --model gpt2-small` 确认 Inshrinkerator baseline (first-order + uniform) 正常工作
3. 确认 `run_inshrinkerator_like_search.py` 搜索输出为单一全局 F 值
