# 项目重构与对比实验集成建议

> 基于论文 EMNLP_26.tex、实验计划 experiment_plan.md、以及对全部源码的深度分析

## 一、核心问题诊断

### 1.1 ExCP / Inshrinkerator 未接入对比实验

现有 4 个对比实验脚本（`run_main_experiment.py`, `run_ablation.py`, `run_pareto.py`, `run_fault_tolerant.py`）只对比了自己方法的变体（magnitude / first-order / second-order × uniform / gamma-adaptive），**完全没有集成 ExCP 和 Inshrinkerator 作为 baseline**。

这意味着论文 Table 1 中计划的 4 行对比（Magnitude / Inshrinkerator / ExCP / Ours）目前无法产出。

### 1.2 接口不统一

三个方法的接口完全不同，无法用统一流程调用：

| 方法 | 接口 | 输入 | 输出 |
|------|------|------|------|
| ExCP | `compress(W_t, O_t, prev_W_hat)` | 权重 + 优化器状态 + 前一重建权重 | `bytes` |
| Inshrinkerator | `compress(W_t, grad, prev_quantized)` | 权重 + 梯度 + 前一量化索引 | `bytes` |
| AdamPrune | 无 compress 接口 | 权重 + 梯度 + exp_avg_sq → 得分 → 剪枝 | 剪枝后的权重 |

关键差异：
- ExCP 做**残差剪枝**（需要前一检查点），其他方法做**直接剪枝**
- Inshrinkerator 需要**梯度**来计算一阶敏感度
- AdamPrune 需要**梯度 + Adam 二阶矩**来计算 damage score
- ExCP 和 Inshrinkerator 包含**量化**步骤，AdamPrune 只做剪枝

### 1.3 AdamPrune 缺少压缩器封装

AdamPrune 是一个研究工具集（importance → distribution → pruning），没有封装成 `BaseCompressor` 子类。论文方法需要一个端到端的压缩器来参与对比实验。

### 1.4 论文与代码的 Gamma vs Weibull 不一致

- **论文**（EMNLP_26.tex）：使用 **Gamma 分布**拟合 damage score，矩估计法
- **AdamPrune 核心代码**（`layer_pruning.py`, `distribution_fast.py`）：使用 **Weibull 分布**拟合，线性回归法
- **对比实验脚本**（`run_main_experiment.py` 等）：使用 **Gamma 分布**（scipy.stats.gamma.fit）

这个不一致需要统一。论文写的是 Gamma，实验脚本也用 Gamma，但底层核心模块用的是 Weibull。建议以论文为准，统一使用 Gamma。

### 1.5 对比实验的公平性问题

实验计划要求"所有方法在相同全局剪枝比例下对比"，但：
- ExCP 的剪枝是在**残差**上做的（`ΔW = W_t - Ŵ_{t-1}`），不是直接剪枝权重
- Inshrinkerator 的剪枝率是通过 `prune_fraction` 配置的（默认 20%），不是全局稀疏率
- 两者都包含量化步骤，而 AdamPrune 只做剪枝

需要明确对比维度：是对比**剪枝策略**（只比剪枝部分），还是对比**端到端压缩**（包含量化）。

---

## 二、建议的对比实验设计

### 2.1 对比维度拆分

建议分两个层面对比：

**层面 A：剪枝策略对比（Table 1 / Table 3 / Fig 4）**

只比较剪枝部分，不涉及量化。所有方法在相同全局稀疏率下，直接对权重做剪枝，评估剪枝后的 loss/perplexity。

| 方法 | 重要性得分 | 率分配 |
|------|-----------|--------|
| Magnitude-Uniform | `|w_i|` | 全局统一阈值 |
| Inshrinkerator-style | `|g_i · w_i|`（一阶） | 全局统一阈值 |
| ExCP-style | `|Δw_i|`（残差 magnitude） | 全局统一阈值 |
| Ours-1st-Uniform | `|g_i · w_i|` | Uniform |
| Ours-2nd-Uniform | `|-g_i·w_i + 0.5·h_i·w_i²|` | Uniform |
| Ours-1st-Gamma | `|g_i · w_i|` | Gamma-adaptive |
| **Ours-Full** | `|-g_i·w_i + 0.5·h_i·w_i²|` | **Gamma-adaptive** |

这样做的好处：
- 公平对比：所有方法在相同稀疏率下直接剪枝
- 消融清晰：Table 3 可以直接从这些组合中提取
- 不需要修改 ExCP/Inshrinkerator 的压缩器代码

**层面 B：端到端压缩对比（Table 2 / Fig 6）**

比较完整的压缩流程（剪枝 + 量化 + 编码），评估压缩比和恢复后的训练质量。

| 方法 | 完整流程 |
|------|---------|
| ExCP | 残差编码 + 联合剪枝 + K-means 量化 + int4 打包 |
| Inshrinkerator | 三向分区 + DDSketch K-means + RLE 增量编码 |
| **Ours + 4bit quant** | Damage score 剪枝 + Gamma 分配 + 4bit 量化 |

### 2.2 ExCP 残差剪枝的特殊处理

ExCP 的核心是残差剪枝（`ΔW = W_t - W_{t-1}`），这与直接剪枝是不同的范式。在层面 A 的对比中，有两种处理方式：

**方案 1（推荐）：提取 ExCP 的重要性度量，用统一框架对比**

把 ExCP 的重要性度量理解为"残差 magnitude"：`|ΔW_i|`。在对比实验中：
1. 计算残差 `ΔW = W_current - W_previous`
2. 用 `|ΔW_i|` 作为重要性得分
3. 用统一的全局阈值剪枝
4. 评估剪枝后的 loss

这需要有前一个检查点。在容错训练实验（Fig 6）中自然满足。在静态剪枝实验（Table 1）中，可以用预训练权重作为 `W_previous`。

**方案 2：在论文中说明 ExCP 使用不同范式**

在 Table 1 中标注 ExCP 使用残差剪枝，并在文中解释差异。这更诚实但可能被审稿人质疑公平性。

**建议采用方案 1**，在统一框架下对比。

---

## 三、代码修改建议

### 3.1 新增统一剪枝接口（最高优先级）

创建一个统一的剪枝接口，封装所有方法的重要性得分计算：

```
src/ckpt_compress/methods/pruning_interface.py
```

```python
class PruningMethod:
    """统一剪枝接口，封装不同方法的重要性得分计算。"""
    
    def compute_scores(
        self,
        weights: Dict[str, Tensor],
        gradients: Optional[Dict[str, Tensor]] = None,
        exp_avg_sq: Optional[Dict[str, Tensor]] = None,
        prev_weights: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """计算每个参数的重要性得分。"""
        ...
    
    def prune(
        self,
        model: nn.Module,
        scores: Dict[str, Tensor],
        prune_ratio: float,
        allocation: str = 'uniform',  # 'uniform' | 'gamma-adaptive'
    ) -> Tuple[nn.Module, float]:
        """应用剪枝，返回 (剪枝后模型, 实际剪枝率)。"""
        ...


class MagnitudePruning(PruningMethod):
    """Magnitude-based pruning: |w_i|"""

class FirstOrderPruning(PruningMethod):
    """Inshrinkerator-style: |g_i · w_i|"""

class ResidualMagnitudePruning(PruningMethod):
    """ExCP-style: |ΔW_i| = |W_t - W_{t-1}|"""

class SecondOrderPruning(PruningMethod):
    """Ours: |-g_i·w_i + 0.5·h_i·w_i²|"""
```

### 3.2 重构对比实验脚本

将现有 4 个对比脚本重构为使用统一接口：

```
experiments/scripts/comparison/
├── run_table1.py          # Table 1: 固定剪枝比例下的质量对比（所有方法）
├── run_table2.py          # Table 2: 端到端压缩比对比
├── run_table3_ablation.py # Table 3: 消融实验
├── run_fig3_gamma_fit.py  # Fig 3: Gamma 拟合质量验证
├── run_fig4_pareto.py     # Fig 4: Pareto 曲线
├── run_fig5_layer_rates.py# Fig 5: 层级剪枝率分布
├── run_fig6_fault_tol.py  # Fig 6: 容错训练 loss 曲线
└── common.py              # 共享工具函数
```

关键改动：
- `run_table1.py`：集成 Magnitude / Inshrinkerator-style / ExCP-style / Ours 四种方法
- `run_fig6_fault_tol.py`：集成 ExCP 和 Inshrinkerator 的完整压缩流程
- `common.py`：提取重复代码（梯度收集、批次缓存、评估函数）

### 3.3 统一 Gamma 分布拟合

在 `src/ckpt_compress/methods/adam_prune/` 中统一使用 Gamma 分布：

1. 在 `distribution.py` 中添加 `fit_gamma_moments()` 函数（矩估计法，与论文公式一致）
2. 在 `layer_pruning.py` 中添加 Gamma CDF 二分法求解（与论文 Eq.10 一致）
3. 保留 Weibull 作为备选（放在 supplementary 实验中）

```python
def fit_gamma_moments(scores: Tensor) -> Tuple[float, float]:
    """矩估计法拟合 Gamma 分布（论文 Eq.8）。
    
    k_hat = mean^2 / var
    theta_hat = var / mean
    """
    mean = scores.mean().item()
    var = scores.var().item()
    if var == 0 or mean == 0:
        return None, None
    k_hat = mean ** 2 / var
    theta_hat = var / mean
    return k_hat, theta_hat


def solve_gamma_threshold(layer_params, global_prune_ratio):
    """二分法求解全局阈值 c*（论文 Eq.10）。
    
    Σ n_l · γ(k_l, c/θ_l) / Γ(k_l) = P · Σ n_l
    """
    ...
```

### 3.4 为 AdamPrune 添加压缩器封装（可选，用于 Table 2）

如果需要做端到端压缩比对比（Table 2），需要为 AdamPrune 添加压缩器：

```
src/ckpt_compress/methods/adam_prune/compressor.py
```

```python
class AdamPruneCompressor(BaseCompressor):
    """基于 damage score 的检查点压缩器。
    
    流程: damage score 剪枝 + Gamma 分配 + K-means 量化 + 序列化
    """
    
    def compress(self, state_dict, gradients=None, exp_avg_sq=None, ...):
        # 1. 计算 damage score
        # 2. Gamma 拟合 + 二分法求全局阈值
        # 3. 应用剪枝
        # 4. 对非零值做 K-means 量化
        # 5. 序列化
        ...
    
    def decompress(self, data):
        # 1. 反序列化
        # 2. 反量化
        # 3. 重建权重
        ...
```

### 3.5 提取共享工具函数

现有 4 个对比脚本有大量重复代码（梯度收集、批次缓存、评估、Gamma 拟合、二分法）。提取到共享模块：

```
experiments/scripts/comparison/common.py
```

包含：
- `cache_batches()` — 缓存数据批次
- `collect_gradients_and_momentum()` — 累积梯度和 Adam 状态
- `filter_prunable_params()` — 过滤可剪枝参数
- `evaluate_loss()` — 评估模型 loss
- `fit_gamma_distribution()` — Gamma 拟合
- `solve_global_threshold()` — 二分法求解
- `apply_uniform_pruning()` — Uniform 剪枝
- `apply_gamma_pruning()` — Gamma-adaptive 剪枝
- `load_model_and_data()` — 加载模型和数据
- `save_results()` — 保存 CSV/JSON 结果

### 3.6 支持更多模型（按实验计划优先级）

现有对比脚本只支持 GPT-2 Small/Medium。需要扩展支持：

| 优先级 | 模型 | 数据集 | 需要修改 |
|--------|------|--------|---------|
| P0 | GPT-2 Medium | WikiText-103 | 已支持，需验证 |
| P2 | BERT-Large | SST-2/MNLI/STS-B | 需要添加 BERT 评估逻辑 |
| P3 | ViT-L/32 | ImageNet | 需要添加 CV 评估逻辑 |

在 `common.py` 中添加模型/数据集的统一加载和评估接口。

---

## 四、ExCP 和 Inshrinkerator 对比的具体设置

### 4.1 层面 A：剪枝策略对比

**Inshrinkerator 的集成方式：**

Inshrinkerator 的核心剪枝策略是一阶敏感度 `|g_i · w_i|`。在统一框架中：
1. 使用现有的 `compute_importance_scores_first_order()` 计算得分（这与 Inshrinkerator 的 `sensitivity()` 函数等价）
2. 用全局统一阈值剪枝（而非 Inshrinkerator 原始的三向分区）
3. 这样对比的是**重要性度量**的质量，而非完整压缩流程

**ExCP 的集成方式：**

ExCP 的核心剪枝策略是残差 magnitude。在统一框架中：
1. 计算残差 `ΔW = W_current - W_reference`
   - 静态实验（Table 1）：`W_reference` = 预训练权重
   - 容错实验（Fig 6）：`W_reference` = 上一个恢复点的权重
2. 用 `|ΔW_i|` 作为重要性得分
3. 用全局统一阈值剪枝

**新增重要性得分函数：**

```python
# 在 importance.py 中添加
def compute_importance_scores_residual(
    weights: Dict[str, Tensor],
    reference_weights: Dict[str, Tensor],
) -> Dict[str, Tensor]:
    """ExCP-style: 残差 magnitude |W_t - W_ref|"""
    scores = {}
    for name in weights:
        if name in reference_weights:
            scores[name] = torch.abs(weights[name] - reference_weights[name])
        else:
            scores[name] = torch.abs(weights[name])
    return scores
```

### 4.2 层面 B：端到端压缩对比

**ExCP 端到端流程：**
1. 加载当前检查点和前一重建检查点
2. 调用 `ExCPCompressor.compress(W_t, O_t, prev_W_hat)` 得到压缩字节
3. 调用 `ExCPCompressor.decompress(compressed, prev_W_hat)` 恢复权重
4. 用恢复的权重继续训练

**Inshrinkerator 端到端流程：**
1. 加载当前检查点，计算梯度
2. 调用 `InshrinkeratorCompressor.compress(W_t, grad)` 得到压缩字节
3. 调用 `InshrinkeratorCompressor.decompress(compressed)` 恢复权重
4. 用恢复的权重继续训练

**Ours 端到端流程：**
1. 加载当前检查点，计算梯度和 Adam 状态
2. 计算 damage score → Gamma 拟合 → 二分法求阈值 → 剪枝
3. 对非零值做 4-bit 量化
4. 序列化 → 反序列化 → 恢复权重
5. 继续训练

**注意事项：**
- ExCP 需要存储前一重建检查点（额外内存开销）
- Inshrinkerator 需要梯度（需要额外一次前向+反向传播）
- 压缩比的计算需要统一：`原始大小 / 压缩后字节数`

### 4.3 容错训练实验（Fig 6）的方法集成

这是最关键的实验，需要在 `run_fig6_fault_tol.py` 中集成所有方法：

```python
methods = {
    'none':                  # Oracle baseline（不压缩）
    'magnitude-uniform':     # Magnitude + Uniform
    'inshrinkerator':        # 一阶敏感度 + Uniform（Inshrinkerator 的剪枝策略）
    'excp-residual':         # 残差 magnitude + Uniform（ExCP 的剪枝策略）
    'ours-full':             # 二阶 damage score + Gamma-adaptive
}
```

每个方法在恢复点执行：
1. 计算重要性得分（各方法不同）
2. 应用剪枝（Uniform 或 Gamma-adaptive）
3. 继续训练

---

## 五、修改优先级和执行顺序

### Phase 1：基础设施（1-2 天）

1. **创建 `common.py`**：提取共享工具函数，消除代码重复
2. **统一 Gamma 拟合**：在核心模块中添加矩估计法 Gamma 拟合，与论文一致
3. **添加残差重要性得分**：在 `importance.py` 中添加 `compute_importance_scores_residual()`

### Phase 2：对比实验集成（2-3 天）

4. **重构 `run_table1.py`**：集成 4 种方法（Magnitude / Inshrinkerator / ExCP / Ours）
5. **重构 `run_fig6_fault_tol.py`**：集成容错训练的完整对比
6. **重构 `run_table3_ablation.py`**：消融实验（5 种组合）

### Phase 3：补充实验（2-3 天）

7. **添加 BERT-Large 支持**：在对比脚本中支持 BERT + GLUE 任务
8. **端到端压缩对比**（Table 2）：集成 ExCP/Inshrinkerator 的完整压缩流程
9. **可视化脚本**：Fig 3（Gamma 拟合）、Fig 5（层级剪枝率）

### Phase 4：运行实验（3-5 天）

10. **微调检查点**：GPT-2 Medium on WikiText-103（如果现有检查点不够）
11. **运行 Table 1**：所有方法 × 所有稀疏率
12. **运行 Fig 6**：容错训练模拟
13. **运行消融和补充实验**

---

## 六、文件变更清单

### 新增文件

| 文件 | 用途 |
|------|------|
| `src/ckpt_compress/methods/pruning_interface.py` | 统一剪枝接口 |
| `experiments/scripts/comparison/common.py` | 共享工具函数 |
| `experiments/scripts/comparison/run_table1.py` | Table 1 主实验（替代 run_main_experiment.py） |
| `experiments/scripts/comparison/run_table2.py` | Table 2 端到端压缩比 |
| `experiments/scripts/comparison/run_table3_ablation.py` | Table 3 消融（替代 run_ablation.py） |
| `experiments/scripts/comparison/run_fig4_pareto.py` | Fig 4 Pareto 曲线（替代 run_pareto.py） |
| `experiments/scripts/comparison/run_fig6_fault_tol.py` | Fig 6 容错训练（替代 run_fault_tolerant.py） |
| `src/ckpt_compress/methods/adam_prune/compressor.py` | AdamPrune 压缩器封装（可选） |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/ckpt_compress/methods/adam_prune/importance.py` | 添加 `compute_importance_scores_residual()` |
| `src/ckpt_compress/methods/adam_prune/distribution.py` | 添加 `fit_gamma_moments()` 矩估计法 |
| `src/ckpt_compress/methods/adam_prune/layer_pruning.py` | 添加 Gamma CDF 二分法（与论文一致） |

### 可删除/归档文件

| 文件 | 原因 |
|------|------|
| `experiments/scripts/comparison/run_main_experiment.py` | 被 run_table1.py 替代 |
| `experiments/scripts/comparison/run_ablation.py` | 被 run_table3_ablation.py 替代 |
| `experiments/scripts/comparison/run_pareto.py` | 被 run_fig4_pareto.py 替代 |
| `experiments/scripts/comparison/run_fault_tolerant.py` | 被 run_fig6_fault_tol.py 替代 |

---

## 七、论文实验部分的建议结构

```latex
\section{Experiments}

\subsection{Setup}
% 模型、数据集、训练配置、评估指标
% 说明 ExCP 和 Inshrinkerator 的重要性度量如何在统一框架下对比

\subsection{Main Results (Table 1)}
% 固定剪枝比例下的质量对比
% 4 种方法 × 4 个稀疏率 × 2+ 个模型

\subsection{Ablation Study (Table 3)}
% 拆解二阶信息和分布放松的独立增益
% 5 种组合

\subsection{Analysis}
% Fig 3: Gamma 拟合质量
% Fig 4: Pareto 曲线
% Fig 5: 层级剪枝率分布

\subsection{Fault-Tolerant Training (Fig 6)}
% 多次恢复的累积误差对比
% 这是最有说服力的实验

\subsection{End-to-End Compression (Table 2)}
% 联合压缩比对比（剪枝 + 量化）
% 运行时开销对比 (Table 4)
```

---

## 八、风险和注意事项

1. **ExCP 残差剪枝的公平性**：在 Table 1 中用残差 magnitude 作为重要性得分，需要在论文中说明这是对 ExCP 核心思想的提取，而非其完整流程。

2. **Inshrinkerator 的三向分区**：原始 Inshrinkerator 不是简单的全局阈值剪枝，而是三向分区（保护 + 剪枝 + 量化）。在统一框架中只提取其剪枝策略，需要说明。

3. **量化的公平性**：Table 2 中，ExCP 和 Inshrinkerator 使用各自的量化方案，Ours 需要选择一个量化方案。建议使用与 Inshrinkerator 相同的 K-means 量化（16 级），保持公平。

4. **检查点可用性**：目前只有 GPT-2 Small 和 BERT-Large 的微调检查点，缺少 GPT-2 Medium 的检查点。需要先运行微调。

5. **Gamma 拟合失败**：某些层（如 LayerNorm）的 damage score 分布可能不服从 Gamma。需要 fallback 机制（如使用经验 CDF）。

6. **论文中 Gamma vs 代码中 Weibull**：必须在提交前统一。建议以论文为准用 Gamma，Weibull 结果放附录作为 robustness check。
