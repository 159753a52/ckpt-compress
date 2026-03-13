# Inshrinkerator-like 剪枝比例搜索（实现计划）

目标：在不依赖 Inshrinkerator 未开源代码的情况下，复现其"按 layer type 施加剪枝比例 + 配置搜索"的核心思想，产出一套可运行的 baseline，用于与我们的 Gamma-adaptive 分配策略作对比。

> 说明（诚实口径）：本计划只覆盖 **pruning ratio / allocation** 相关逻辑，刻意不去复现其完整系统（保护比例、非均匀量化、delta 编码、参数重排等）。因此该 baseline 的效果可能明显弱于论文中的完整 Inshrinkerator；但它能作为"一阶敏感性 + per-layer-type 搜索"的对照。论文中需注明这一点。

---

## 1. 从论文抽取的关键设计点（我们要对齐的部分）

来自 `paper/Inshrinkerator_2306.11800v3_tex/content/quantization.tex` 和 `search.tex`：

1) 重要性打分：
- Magnitude：I_m(w) = |w|
- Sensitivity（一阶泰勒近似）：I_s(w) = |g(w) * w|
- 论文用训练过程中的梯度 EMA（beta=0.9，50 个 batch）估计 sensitivity

2) 剪枝（pruning）：
- 论文描述的剪枝比例 F_prun 通常在 10%~40%
- **per-layer-type pruning**：同一种 layer type 的所有层，使用同一剪枝比例（阈值按该 type 内分位数/quantile 定）

3) 搜索（search）：
- 论文完整系统会在"剪枝比例、保护比例、量化 bins"等多维配置上做 guided search / delta-neighborhood search
- **关键细节**：论文对 magnitude 和 sensitivity 两种 metric 分别执行搜索，然后选更优的那个（search.tex: "We repeat this search procedure with the two pruning metrics, magnitude and sensitivity, and select the better of the two."）

我们本次只实现：
- pruning metric（magnitude vs sensitivity）的选择（两种都搜，取更优）
- layer-type 粒度上的 pruning ratio 分配与搜索

### 1.1 梯度估计差异说明

论文用 EMA 梯度（beta=0.9，50 batch），我们现有 `compute_scores_by_method` 里 `first-order` 用的是累积平均梯度（`compute_gradients_batched`，默认 50 步）。两者都是梯度的平滑估计，差异不大。论文中注明："我们用 50 步累积平均梯度近似 Inshrinkerator 的 EMA 梯度"即可，不需要改代码。

---

## 2. 我们的 baseline 定义（Inshrinkerator-like Pruning Search）

### 2.1 layer type 的定义

对于 Transformer（GPT-2 / BERT）用"参数名规则"做粗分类：

- `embed`：embedding 相关（不剪）
- `attn`：attention 线性层权重（Q/K/V/O、或 fused qkv）
- `mlp`：MLP/FFN 线性层权重
- `others_linear`：其它 2D Linear 权重（如 lm_head / pooler 等，视情况）
- `norm/bias/1D`：LayerNorm、bias、以及所有 1D 参数（不剪）

对于 ResNet：
- `conv`
- `fc`
- `bn/bias/1D`（不剪）

落地形式：实现一个 `infer_layer_type(param_name, tensor) -> str`，把 prunable 的 named_parameters 映射到 type。

### 2.2 剪枝算子（per-layer-type 分位数阈值）

给定某个 type t 的所有得分集合 score_t（把该 type 下所有 tensor flatten 后拼一起）：

- 目标剪枝比例 p_t（0~1）
- 阈值 c_t = quantile(score_t, q=p_t)
- 该 type 内 mask 规则：mask = 1(score > c_t)

重要：同一 type 的所有层共享同一个阈值 c_t（这是"per-layer-type pruning"的关键——同 type 内按全局分位数剪，而非逐层分位数）。

### 2.3 质量约束（搜索目标）

统一判定公式：

- LM：ppl_drop_pct = (ppl_pruned - ppl_base) / ppl_base
- CLS/CV：acc_drop = (acc_base - acc_pruned) / acc_base
- REG：pearson_drop = (pearson_base - pearson_pruned) / max(|pearson_base|, eps)

约束：drop <= epsilon。

建议同时报告两个阈值的结果：
- epsilon=0.01（1%，保守，我们偏好的设定）
- epsilon=0.05（5%，论文默认设定，更公平的对比）

---

## 3. 搜索方案：per-layer-type ratio 搜索（方案 B）

采用 per-type-specific ratio 搜索，能直接产出不同 type 的剪枝比例，便于与 Gamma-adaptive 做可视化对比。

### 3.1 搜索空间

配置包含：
- pruning_metric ∈ {magnitude, sensitivity}
- p_attn ∈ 候选集（见下）
- p_mlp ∈ 候选集（见下）
- （可选）p_others_linear ∈ {0, 0.1, 0.2}

### 3.2 两阶段搜索策略

为减少评估次数（每次评估需要前向推理），采用粗→细两阶段搜索：

**阶段 1：粗网格定位**
- p_attn, p_mlp ∈ {0, 0.2, 0.4}
- 3×3 = 9 个候选
- 对 magnitude 和 sensitivity 分别搜索 → 共 18 次评估
- 按"预计全局剪枝率"从大到小排序，依次评估，找到第一个满足约束的配置

**阶段 2：细网格精搜**
- 在阶段 1 最优配置的邻域内，用 step=0.1 细搜
- 例如阶段 1 最优为 (p_attn=0.2, p_mlp=0.4)，则细搜范围：
  - p_attn ∈ {0.1, 0.2, 0.3}
  - p_mlp ∈ {0.3, 0.4, 0.5}
- 3×3 = 9 个候选，对当前最优 metric 搜索 → 9 次评估

总评估次数：~27 次（比原始 36 次少，且精度更高）。

### 3.3 metric 选择

遵循论文做法：对 magnitude 和 sensitivity 分别执行搜索，最终选择"在满足约束的前提下，全局剪枝率更高"的那个 metric 及其对应的 per-type ratio。若全局剪枝率相同，优先 sensitivity（更贴近论文）。

### 3.4 搜索目标

在质量约束（drop <= epsilon）下，最大化全局被剪掉的参数数（等价最大化 global pruning ratio）。

全局剪枝率的预估公式（用于排序候选，无需实际评估）：
```
global_ratio_est = Σ_t (n_t * p_t) / Σ_t n_t
```
其中 n_t 是 type t 的参数总数。

---

## 4. 与现有代码的集成

### 4.1 新增文件

1) 搜索逻辑（独立模块）：
- `src/ckpt_compress/pruning/inshrinkerator_search.py`
  - `infer_layer_type(name: str, tensor: Tensor) -> str` — 参数名→type 映射
  - `apply_pruning_per_type(model, scores, per_type_ratios, type_map) -> (model, masks, actual_ratio_by_type, actual_global_ratio)` — 按 type 分位数剪枝
  - `search_best_config(model_factory, scores, type_map, cached_eval, task_type, device, epsilon, ...) -> SearchResult` — 两阶段搜索
  - `SearchResult` dataclass：包含 best_metric, best_per_type_ratios, actual_global_ratio, quality_metrics

2) 搜索结果持久化 + 读取分配策略：
- `src/ckpt_compress/pruning/per_type_allocation.py`
  - `PerTypeAllocation(AllocationStrategy)` — 从 JSON 文件读取预搜索的 per-type ratio，在 `allocate()` 时按 type 映射到每个参数
  - 注册到 `ALLOCATION_REGISTRY["per-type"]`

3) 搜索脚本：
- `experiments/scripts/run_inshrinkerator_like_search.py`
  - 输入：model/dataset/checkpoint、epsilon（支持多个，如 0.01,0.05）、候选 ratio 范围
  - 输出：JSON 文件（per-type ratio + 搜索元信息），保存到 `results/paper_results/inshrinkerator_search/`

### 4.2 JSON 结果格式

搜索脚本输出的 JSON 格式：

```json
{
  "model": "gpt2-small",
  "dataset": "wikitext103",
  "checkpoint": "checkpoints/gpt2_small_wikitext103_1000steps/...",
  "epsilon": 0.01,
  "best_metric": "sensitivity",
  "per_type_ratios": {
    "attn": 0.3,
    "mlp": 0.4,
    "others_linear": 0.1
  },
  "actual_global_ratio": 0.347,
  "baseline_metrics": {"loss": 3.21, "perplexity": 24.8},
  "pruned_metrics": {"loss": 3.24, "perplexity": 25.4},
  "quality_drop_pct": 0.93,
  "search_log": [...]
}
```

### 4.3 Table1 集成方式

在 Table1 中，Inshrinkerator-like baseline 作为一种 allocation 策略参与对比：

1. 先运行 `run_inshrinkerator_like_search.py`，产出 JSON
2. Table1 脚本中，用 `PerTypeAllocation` 读取 JSON，作为 `allocation='per-type'` 参与对比
3. `PerTypeAllocation.allocate(scores, global_prune_ratio)` 的行为：
   - 读取 JSON 中的 per-type ratio
   - 按比例缩放到目标 global_prune_ratio（保持各 type 间的比例关系不变）
   - 返回 {param_name: scaled_ratio}

缩放公式：
```
scale = target_global_ratio / json_global_ratio
p_t_scaled = min(p_t * scale, 0.95)  # 上限保护
```

这样 Inshrinkerator-like baseline 就能在 Table1 的多个 target_ratio 下和 uniform、gamma-adaptive 公平对比。

### 4.4 METHODS 列表更新

Table1 的 METHODS 新增一行：
```python
{'importance': 'first-order', 'allocation': 'per-type'},  # Inshrinkerator-like
```

注意：这里 importance 固定为搜索结果中的 best_metric 对应的方法（magnitude→'magnitude'，sensitivity→'first-order'）。

---

## 5. 预期产出

最小可交付：
- 一组 Inshrinkerator-like 配置（metric + per-type ratios），以 JSON 持久化
- 对应的：baseline 指标、剪枝后指标、退化百分比、实际 global pruning ratio
- Table1 中新增一行 Inshrinkerator-like baseline 的结果

可选增强：
- 把 per-type ratios 画成柱状图（与 Gamma-adaptive 的逐层率对比）
- 在 epsilon=0.01 和 0.05 两个阈值下分别报告搜索结果

---

## 6. 风险与局限

1) **系统 vs 组件**：论文的 Inshrinkerator 是完整系统（保护、非均匀量化、delta 编码、参数重排、动态配置搜索），我们只做 pruning ratio/search，效果可能偏弱。论文中需明确说明。

2) **type 定义依赖命名规则**：不同模型（GPT-2/BERT/ResNet）需要单独校准 `infer_layer_type` 的规则。

3) **梯度估计差异**：论文用 EMA 梯度（beta=0.9），我们用累积平均梯度。差异不大，但需在论文中注明。

4) **单调性假设**：搜索策略假设"更多剪枝→质量单调变差"。一般成立，但少量点可能有噪声。缓解措施：
   - 找到可行解后，继续向更激进方向试 1~2 步确认边界
   - 可选：同一候选重复评估 2 次取均值

5) **缩放近似**：Table1 中用比例缩放将搜索结果映射到不同 target_ratio，这是近似做法。在极端 ratio（如 0.9）下可能不准确。但作为 baseline 足够。

---

## 7. 可维护性与扩展性设计（代码修改建议）

本节针对"如何写代码才能方便后续扩展新模型/新数据集"给出具体建议。核心原则：**新增模型或数据集时，只需要在一个地方注册，不需要改动多个文件的 if/elif 链。**

### 7.1 问题：参数名解析逻辑散落在多处

当前代码中，"参数名 → 类别"的映射逻辑存在于两个地方，且即将出现第三个：

| 位置 | 用途 | 粒度 |
|------|------|------|
| `run_fig5_heatmap.py` 的 `parse_gpt2_param()` / `parse_bert_param()` / `parse_resnet_param()` | 热力图：param → (block_id, param_group) | 细粒度（区分 Q/K/V/O/FC/Proj/LN） |
| `pruner.py` 的 `filter_prunable_params()` | 过滤不可剪枝参数 | 粗粒度（只判断 embed/bias/norm） |
| **本计划** 的 `infer_layer_type()` | Inshrinkerator 搜索：param → layer_type | 中粒度（区分 attn/mlp/others） |

三处逻辑高度重叠，但各自独立维护。新增模型（如 LLaMA、ViT）时需要同时改三个地方。

**建议：抽取统一的 `ModelParamSchema`，放在 `src/ckpt_compress/pruning/param_schema.py`。**

```python
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Tuple

@dataclass
class ParamInfo:
    """单个参数的结构化元信息。"""
    name: str               # 原始参数名
    layer_type: str          # 粗粒度: 'attn' | 'mlp' | 'conv' | 'fc' | 'embed' | 'norm' | 'other'
    param_group: str         # 细粒度: 'attn_qkv' | 'attn_proj' | 'mlp_fc' | 'mlp_proj' | 'ln_1' | ...
    block_id: Optional[int]  # Transformer block 编号，非 block 结构为 None
    prunable: bool           # 是否可剪枝


class ModelParamSchema:
    """模型参数命名规则的抽象。每个模型族注册一个实例。"""

    def __init__(self, parse_fn: Callable[[str, 'torch.Tensor'], ParamInfo]):
        self._parse_fn = parse_fn

    def parse(self, name: str, tensor: 'torch.Tensor') -> ParamInfo:
        return self._parse_fn(name, tensor)

    def classify_all(self, named_params) -> Dict[str, ParamInfo]:
        return {name: self.parse(name, p) for name, p in named_params}


# --- 注册表 ---
PARAM_SCHEMA_REGISTRY: Dict[str, ModelParamSchema] = {}

def register_param_schema(model_family: str, schema: ModelParamSchema):
    PARAM_SCHEMA_REGISTRY[model_family] = schema

def get_param_schema(model_name: str) -> ModelParamSchema:
    """根据模型名匹配 schema。如 'gpt2-small' → 匹配 'gpt2'。"""
    for family, schema in PARAM_SCHEMA_REGISTRY.items():
        if family in model_name:
            return schema
    raise ValueError(f"No param schema for model: {model_name}")
```

然后为每个模型族注册一个 parse 函数：

```python
def _parse_gpt2(name: str, tensor) -> ParamInfo:
    parts = name.split('.')
    # ... 解析逻辑（合并现有 parse_gpt2_param + infer_layer_type 的 GPT-2 部分）
    # 同时输出 layer_type（粗）和 param_group（细）和 block_id
    # prunable = (layer_type in ('attn', 'mlp', 'other_linear') and tensor.dim() >= 2)
    ...

register_param_schema('gpt2', ModelParamSchema(_parse_gpt2))
register_param_schema('bert', ModelParamSchema(_parse_bert))
register_param_schema('resnet', ModelParamSchema(_parse_resnet))
```

**收益：**
- `infer_layer_type()` → `get_param_schema(model_name).parse(name, tensor).layer_type`
- `filter_prunable_params()` → `{name: w for name, w in weights.items() if schema.parse(name, w).prunable}`
- `run_fig5_heatmap.py` 的 `parse_gpt2_param()` → `schema.parse(name, tensor).block_id, schema.parse(name, tensor).param_group`
- 新增模型只需写一个 `_parse_xxx()` 函数并 `register_param_schema()`，所有下游（搜索、热力图、过滤）自动生效

### 7.2 问题：质量退化判定逻辑将被多处需要

搜索需要 `compute_quality_drop(baseline, pruned, task_type) -> float`，但这个逻辑在 `run_table1.py`、`run_table3_ablation.py` 里也有（以内联形式）。

**建议：在 `experiments/lib/evaluation.py` 中新增一个共享函数。**

```python
def compute_quality_drop(baseline_metrics: Dict, pruned_metrics: Dict, task_type: str) -> float:
    """计算相对质量退化百分比。
    
    返回值 >= 0 表示退化，< 0 表示改善。
    """
    if task_type == 'lm':
        return (pruned_metrics['perplexity'] - baseline_metrics['perplexity']) / baseline_metrics['perplexity']
    elif task_type in ('cls', 'cv'):
        return (baseline_metrics['accuracy'] - pruned_metrics['accuracy']) / max(baseline_metrics['accuracy'], 1e-10)
    elif task_type == 'reg':
        return (baseline_metrics['pearson'] - pruned_metrics['pearson']) / max(abs(baseline_metrics['pearson']), 1e-10)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")
```

搜索中直接调用：`drop = compute_quality_drop(baseline, pruned, task_type); feasible = (drop <= epsilon)`。

### 7.3 问题：`get_data_loaders()` 和 `load_model()` 的 if/elif 链

当前 `experiments/lib/data.py` 的 `get_data_loaders()` 和 `experiments/lib/models.py` 的 `load_model()` 都用 if/elif 链分发。新增数据集或模型需要改动这些核心函数。

**本次不做大改**（改动面太大，与 Inshrinkerator 搜索无关），但建议后续重构方向：

- `data.py`：改为注册表模式，类似 `MODEL_REGISTRY`：
  ```python
  DATA_REGISTRY = {
      'wikitext103': {'factory': _make_wikitext103_loaders, 'task_type': 'lm'},
      'sst2':        {'factory': _make_sst2_loaders,        'task_type': 'cls'},
      ...
  }
  ```
  新增数据集只需 `DATA_REGISTRY['new_dataset'] = {...}`。

- `models.py`：`MODEL_REGISTRY` 已经存在，但 BERT 的特殊逻辑（GLUE head 适配）仍然是内联 if/elif。建议把 BERT 的加载逻辑封装成一个 factory 函数注册进去，消除 `load_model()` 中的特殊分支。

**本次实现中的应对**：`inshrinkerator_search.py` 不直接依赖 `get_data_loaders()` 或 `load_model()`，它只接收已经准备好的 `scores`、`model_factory`、`cached_eval`。这样搜索逻辑本身不受模型/数据集扩展影响。

### 7.4 问题：搜索网格的硬编码

两阶段搜索的候选值（`{0, 0.2, 0.4}` 等）如果硬编码在 `search_best_config()` 里，后续调整不方便。

**建议：用 dataclass 封装搜索配置。**

```python
@dataclass
class SearchConfig:
    """搜索超参数，与模型/数据集无关。"""
    coarse_candidates: List[float] = field(default_factory=lambda: [0.0, 0.2, 0.4])
    fine_step: float = 0.1
    fine_radius: int = 1          # 邻域半径（±1 步）
    metrics: List[str] = field(default_factory=lambda: ['magnitude', 'sensitivity'])
    epsilons: List[float] = field(default_factory=lambda: [0.01, 0.05])
    max_type_ratio: float = 0.95  # 单 type 剪枝率上限
```

搜索脚本中可以从 YAML 加载，也可以用默认值：

```python
# 默认
cfg = SearchConfig()
# 或从 YAML
cfg = SearchConfig(coarse_candidates=[0, 0.15, 0.3, 0.45], fine_step=0.05)
```

### 7.5 问题：新增模型时 `infer_layer_type` 的扩展路径

即使不做 7.1 的完整重构，`infer_layer_type()` 本身也需要设计成可扩展的。

**建议：用 model_family 参数 + 内部分发，而非在函数体内写一个巨大的 if/elif。**

```python
# 每个模型族的规则是一个独立函数
_TYPE_RULES: Dict[str, Callable[[str, torch.Tensor], str]] = {}

def register_type_rule(family: str, fn: Callable[[str, torch.Tensor], str]):
    _TYPE_RULES[family] = fn

def infer_layer_type(name: str, tensor: torch.Tensor, model_family: str) -> str:
    if model_family not in _TYPE_RULES:
        raise ValueError(f"No type rules for model family: {model_family}. "
                         f"Register with register_type_rule(). Available: {list(_TYPE_RULES.keys())}")
    return _TYPE_RULES[model_family](name, tensor)

# --- 注册 ---
def _gpt2_type_rule(name: str, tensor: torch.Tensor) -> str:
    if tensor.dim() < 2 or any(p in name for p in ['ln_', 'bias', 'wte', 'wpe']):
        return 'skip'
    if 'attn' in name:
        return 'attn'
    if 'mlp' in name:
        return 'mlp'
    return 'others_linear'

register_type_rule('gpt2', _gpt2_type_rule)
register_type_rule('bert', _bert_type_rule)
register_type_rule('resnet', _resnet_type_rule)
```

新增模型（如 LLaMA）时：
```python
def _llama_type_rule(name, tensor):
    if tensor.dim() < 2 or 'norm' in name or 'bias' in name:
        return 'skip'
    if 'self_attn' in name:
        return 'attn'
    if 'mlp' in name:
        return 'mlp'
    return 'others_linear'

register_type_rule('llama', _llama_type_rule)
```

不需要改动 `infer_layer_type()` 本身或搜索逻辑。

### 7.6 总结：本次实现中必须做的 vs 建议后续做的

**本次必须做（直接影响 Inshrinkerator 搜索的可维护性）：**

1. `infer_layer_type()` 用注册表模式（7.5），不要写成一个大 if/elif
2. `compute_quality_drop()` 抽到 `experiments/lib/evaluation.py`（7.2）
3. `SearchConfig` dataclass 封装搜索超参数（7.4）
4. `search_best_config()` 只接收准备好的 scores/model_factory/cached_eval，不直接依赖模型加载或数据加载（7.3 的应对策略）

**建议后续做（更大范围的重构，不阻塞本次）：**

1. 统一 `ModelParamSchema`（7.1），合并 Fig5 的 parse 函数和本次的 type rule
2. `get_data_loaders()` 改为注册表模式（7.3）
3. `load_model()` 消除 BERT 特殊分支（7.3）

---

## 8. 落地顺序

1. 实现 `infer_layer_type()` 注册表 + GPT-2/BERT/ResNet 规则 + 单元测试
2. 实现 `compute_quality_drop()` 并放入 `experiments/lib/evaluation.py`
3. 实现 `SearchConfig` dataclass
4. 实现 `apply_pruning_per_type()` + 单元测试
5. 实现 `search_best_config()`（两阶段搜索）
6. 实现 `PerTypeAllocation` + 注册到 `ALLOCATION_REGISTRY`
7. 编写 `run_inshrinkerator_like_search.py` 脚本
8. 在 GPT-2 Small + WikiText-103 上验证搜索流程
9. 集成到 Table1，确认结果合理
