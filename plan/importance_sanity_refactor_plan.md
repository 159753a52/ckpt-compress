# 一阶 vs 二阶（HVP）重要性对比框架重构方案（修订版）

## 1. 目标变更（按你的要求）

本方案从“单脚本 CLI 实验”改为“**可编排实验框架**”，核心要求如下：

1. **运行方式**：以 Python 框架/配置驱动运行（可被 notebook、主控脚本、调度器调用），不是以 CLI 为中心。
2. **任务覆盖**：支持多模型、多任务统一执行与对比。
3. **二阶信息来源**：**移除 Adam 二阶近似路径**，二阶仅保留 **HVP**。
4. **实验范围**：模型/数据集支持范围以 `plan/experiment_plan.md` 为准（P0~P3 优先级）。
5. **落地约束**：优先复用现有 `experiments/lib/` 与 `src/ckpt_compress/pruning/`，避免新增“平行框架”造成重复实现与维护成本。

---

## 2. 与 experiment_plan.md 对齐的支持范围

## 2.1 目标支持矩阵（论文优先级）

- **P0**：GPT-2 Medium + WikiText-103（LM，指标：loss / ppl）
- **P1**：Pythia-410M + Alpaca（指令微调后任务评估，指标：Avg Acc）
- **P2**：BERT-Large + GLUE（SST-2/MNLI/STS-B）
- **P3**：ViT-L/32 + ImageNet-1K（CV，指标：Top-1）

## 2.2 当前代码现状与落地策略

- 代码库现成支持较完整的是 GPT-2 / BERT / ResNet 与 WikiText/GLUE/CIFAR。
- Pythia 与 ViT-L/32 + ImageNet 需新增 model/data adapter。

因此框架重构分两层：

- **Core 框架先统一完成**（支持已接入模型任务）
- **Adapter 增量扩展**到 Pythia/Alpaca 与 ViT/ImageNet，不改核心流程

---

## 2.3 与当前仓库结构对齐（重要差异与必改点）

> 本节是对现有代码库（2026-02-26 重构版）的对齐审查结论，用于降低实施风险。

### 2.3.1 已存在的“统一能力”，不建议重复造轮子

- 剪枝统一接口已在 `src/ckpt_compress/pruning/`：
  - `importance.py`（magnitude / first-order / second-order / residual-magnitude）
  - `allocation.py`（uniform / gamma-adaptive）
  - `pruner.py`（过滤+应用剪枝）
- 实验共享库已在 `experiments/lib/`：
  - `models.py` / `data.py` / `gradient.py` / `evaluation.py` / `results.py`
- 论文脚本已基本“薄封装”形式存在：`experiments/scripts/run_table1.py`、`run_table3_ablation.py`、`run_fig4_pareto.py`、`run_fig5_heatmap.py`、`run_fig6_fault_tolerant.py`

因此本次重构的重点不应是“再做一套 experiments/framework”，而是：

1) **补齐 HVP-only 二阶重要性**（作为新的 importance 方法），并与现有 `Pruner`/allocation 直接兼容；  
2) 把“可编排 runner + 配置 + 公平性校验 + 结果契约”做成可复用组件，让现有 run_*.py 逐步迁移为薄 wrapper。

### 2.3.2 当前实现存在的兼容性坑（会直接影响 P2）

- **BERT + GLUE 目前不可直接跑通**：
  - 现有 `checkpoints/bert_large_*_1000steps/*.pt` 的 `state_dict` 含 `classifier.*`（SequenceClassification）。
  - 但 `experiments/lib/models.py` 当前加载的是 `src/ckpt_compress/models/bert.py`（`BertForMaskedLM`）。
  - 结论：需要新增/切换到 `BertForSequenceClassification`（SST-2/MNLI）与回归 head（STS-B），否则 checkpoint 无法 strict load，loss/metric 也不匹配。
- **STS-B 需要回归指标**：
  - `src/ckpt_compress/utils/data_loader.py` 中 STS-B label 是 `float32`（回归）。
  - 但 `experiments/lib/evaluation.py` 的 `evaluate_cls` 目前是 CE+accuracy（分类）。
  - 结论：P2 包含 STS-B 时，必须新增 `evaluate_regression`（Pearson/Spearman 或至少 MSE+Pearson），并在 runner 中按任务类型选择 loss_fn。

---

## 3. 框架化设计（非 CLI 中心）

## 3.1 推荐落地形态（在现有结构上增量）

为了避免与 `experiments/lib/`、`experiments/scripts/`、`src/ckpt_compress/pruning/` 平行重复，建议将“可编排框架”落在：

`experiments/lib/importance_compare/`（或单文件 `experiments/lib/importance_compare.py`，视规模决定）

并保留一个**可选的 CLI 薄封装**脚本（例如 `experiments/scripts/run_importance_compare.py`），仅做参数解析后调用 runner。

建议模块划分：

- `runner.py`：实验主入口（类接口），对外暴露 `run()` 返回结构化 report
- `config.py`：dataclass 配置定义（可从 dict/yaml 构造）
- `protocol.py`：公平性协议与校验（强约束，失败即报错）
- `scoring.py`：一阶 / 二阶(HVP) score 计算（只负责“从 model+batches → scores dict”）
- `hvp.py`：HVP 后端（优先抽取为通用实现；必要时可复用 `src/ckpt_compress/methods/adam_prune/importance.py` 的逻辑）
- `io.py`：结果落盘（复用 `experiments/lib/results.py`）
- `adapters/`：模型/数据集/任务适配（优先复用 `experiments/lib/models.py` 与 `experiments/lib/data.py`，不足再补）

明确“不做/不重复”的部分：

- **不重复实现剪枝与分配**：统一复用 `src/ckpt_compress/pruning/`（`Pruner` + `allocation.py`）
- **不重复实现评估**：优先复用 `experiments/lib/evaluation.py`（P2 的 STS-B 回归需补一个分支）

## 3.2 核心调用方式（框架接口）

示例（非 CLI）：

```python
from experiments.framework.importance_compare.runner import ImportanceCompareRunner
from experiments.framework.importance_compare.config import ExperimentConfig

cfg = ExperimentConfig(
    model="bert-large",
    dataset="sst2",
    checkpoint="checkpoints/bert_large_sst2_1000steps/checkpoint_step_1000_final.pt",
    methods=["first-order", "second-order-hvp"],
    prune_ratios=[0.5, 0.7, 0.8, 0.9],
    allocation="uniform",
    num_steps=100,
    eval_batches=20,
    hvp_batches=8,
    alpha=0.5,
    seed=42,
    device="cuda",
)

runner = ImportanceCompareRunner(cfg)
report = runner.run()
```

CLI 仅作为可选薄封装（调用上述 API），不是主入口。

---

## 4. 重要性计算设计（仅 HVP 二阶）

## 4.1 方法定义

- 一阶：
  \[
  s_i^{(1)} = |g_i\,\theta_i|
  \]

- 二阶（仅 HVP）：
  \[
  s_i^{(2)} = \left|-g_i\theta_i + \alpha\,\theta_i(H\theta)_i\right|
  \]

说明：
- 不再提供 `adam exp_avg_sq` 近似路径。
- 二阶统一通过 `HVP` 估计 \((H\theta)_i\)。
- 默认使用多 batch 平均 HVP，降低估计方差。

## 4.2 HVP 计算后端

可复用的现有实现（`src/ckpt_compress/methods/adam_prune/importance.py`）已包含：

- `compute_hvp`
- `compute_hvp_batched`
- `compute_importance_scores_hvp_abs`
- `compute_importance_scores_hvp_memory_efficient`

但需要注意两点改造/规范化（否则会影响对比公平性与数值稳定性）：

1. **梯度与 HVP 的 batch 需对齐**  
   现有 `compute_importance_scores_hvp_abs` 里梯度只取 `data_batches[0]`，而 HVP 平均多个 batch；这会让一阶/二阶对比被“梯度噪声”干扰。  
   计划应明确：`g` 与 `Hθ` 都在同一组 `hvp_batches` 上做平均（或都在 `num_steps` 上做平均）。

2. **“memory_efficient 分块”实际上是近似（block-HVP）**  
   若通过把 `vector=θ` 限制在 chunk 内（其他位置置 0）来做 HVP，那么计算的是 `H·θ_chunk`，并非严格的 `H·θ`。  
   该模式应明确命名为 `hvp_mode="block"`（或在文档中声明是 block-diagonal 近似），并标注：**论文结果优先使用 full HVP**，block 仅用于显存受限的快速迭代。

在新框架中建议封装策略（命名更清晰）：

- `hvp_mode="full"`：一次性计算 `H·θ`（推荐，严格一致）
- `hvp_mode="block"`：分块近似（节省显存/时间的权衡，但会改变 score 定义）

补充建议（可选但强烈推荐写入计划）：

- score 计算时将 `model.eval()`（禁用 dropout），保证可重复性与更低方差
- HVP 尽量用 `float32` 关闭 AMP（混精会放大二阶数值误差）
- 对于显存紧张模型，优先尝试“更小 batch/seq_length + checkpointing + 减少 hvp_batches”，而不是默认 block 近似

---

## 5. 实验公平性协议（强约束）

框架层面强制执行以下规则：

1. 同一 checkpoint 起点
2. 同一训练缓存批次用于 score 估计
3. 同一评估缓存批次用于指标评估
4. 同一可剪枝参数集合（统一过滤）
5. 同一 prune ratio 列表
6. 固定随机种子（python/numpy/torch）
7. 输出 target ratio 与 actual ratio

若违反协议，runner 直接报错并终止。

---

## 6. 与论文实验目标的映射

## 6.1 Table 1：固定分配下的重要性得分对比（核心实验）

### 设计思路

**核心目标**：证明二阶 damage score 优于一阶和 magnitude。

**分配策略选择**：
- ~~uniform~~（已删除）：对所有层施加相同剪枝率不合理，不同层敏感度差异大。
- **Inshrinkerator per-type search（采用）**：通过搜索得到 per-type 剪枝比例（attn、mlp 等），作为所有 importance method 的公共分配策略。
  - 理由 1：公平性好——per-type ratios 是独立于我们方法的外部分配，不会偏向任何一种 importance。
  - 理由 2：真实性强——这就是 Inshrinkerator 原论文的做法。
  - 理由 3：代码已有——`src/ckpt_compress/pruning/inshrinkerator_search.py` 已实现完整搜索。

### Table 1 结构

| 行 | 方法名 | 重要性得分 | 分配策略 | 说明 |
|----|--------|-----------|---------|------|
| 1 | Magnitude | \|w_i\| | Inshrinkerator per-type（固定） | 最弱 baseline |
| 2 | Inshrinkerator | \|g_i · w_i\| | Inshrinkerator per-type（固定） | 一阶 baseline（原论文方法） |
| 3 | ExCP | \|w_t - w_{t-1}\| | Inshrinkerator per-type（固定） | 残差 magnitude baseline |
| 4 | Ours (score only) | damage score | Inshrinkerator per-type（固定） | 证明 score > first-order |
| 5 | **Ours (full)** | **damage score** | **Gamma-adaptive** | 完整方法 |

**对比逻辑**：
- Row 1→2→4：相同分配，不同 importance → 证明二阶得分更优
- Row 4 vs Row 5：相同 importance，不同分配 → 证明 Gamma-adaptive > per-type
- Row 2 是 Inshrinkerator 原论文的完整复现

### per-type 剪枝比例获取方式

**搜索配置**：
- 使用 first-order（|g·w|）作为搜索 metric（Inshrinkerator 原论文方法）
- 搜索得到的 per-type ratios 对 first-order 是"最优的"
- 如果 second-order 在相同分配下还能更好，说明得分质量确实更高

**多全局稀疏率的处理**：
- 方案 A：在不同 epsilon 下多次运行搜索，得到不同全局稀疏率的 per-type ratios
- 方案 B（更快）：搜索一次，用 `PerTypeAllocation` 的比例缩放功能缩放到目标稀疏率（代码已支持 proportional scaling）
- 推荐方案 B 用于快速迭代，方案 A 用于最终论文结果

**搜索步骤**：
1. 加载微调后的检查点（如 GPT-2 Medium + WikiText-103）
2. 计算 first-order + magnitude importance scores
3. 运行 `search_best_config()` 得到 per-type ratios
4. 保存搜索结果 JSON
5. 在 run_table1.py 中通过 `--per_type_json` 加载，作为固定分配

### 每个模型/数据集的实验矩阵

对每个 (model, dataset) 组合：
- 目标全局稀疏率：30%, 50%, 70%, 90%
- 指标：loss, perplexity (LM) / accuracy (cls) / Pearson (reg)
- 报告：loss_increase, loss_increase_pct

## 6.2 Table 3 消融

- importance: magnitude vs first-order vs second-order-hvp
- allocation: per-type vs gamma-adaptive
- 二维消融矩阵，分离各因素贡献

## 6.3 Table 1/Fig4/Fig6 的衔接

- Table 1：固定 per-type 分配 + 多 importance 对比
- Fig4：Table 1 数据的 Pareto 曲线可视化（稀疏率 vs 质量）
- Fig6：在容错训练循环中调用同一评分/剪枝模块（复用核心组件）
---

## 7. 结果目录与数据契约

输出目录：`results/importance_compare/<timestamp>/`

固定文件：

- `summary.csv`：方法×剪枝率核心指标
- `pairwise_first_vs_second.csv`：一阶减二阶差值
- `layer_stats.csv`：每层分数统计
- `config.json`：完整配置
- `meta.json`：环境、seed、公平性校验结果

关键判定列：

- `loss_increase = loss_pruned - loss_baseline`
- `delta_first_minus_second = loss_inc_first - loss_inc_second`
- `delta_first_minus_second > 0` ⇒ 二阶(HVP)更优

建议补充一个“完成判定/验收标准”（让 sanity check 不再主观）：

- 在中高剪枝率（建议 ≥50%）下，`delta_first_minus_second` 在多数设置上为正（例如 ≥70% 的 ratio 点为正）
- 对同一设置重复跑 2 次（不同 seed），方向一致（即不被随机性主导）
- 记录并输出 HVP 运行时间/显存峰值（避免“更好但不可用”）

---

## 8. 实施阶段（修订）

## Phase 0：对齐 P2 的基础设施（0.5~1 天）

- 修正 `bert-large` 的模型适配：GLUE 任务使用 `BertForSequenceClassification`
- STS-B 走回归分支（loss + metric），避免用 CE/accuracy 产生伪结果
- 明确每个任务的 `loss_fn`（用于 HVP）与 `evaluate` 指标一致

## Phase A：框架骨架（0.5~1 天）

- 搭建 `experiments/lib/importance_compare` 模块（runner/config/protocol）
- 将现 `experiments/lib` 的 model/data/eval 能力接入 runner（保持复用）
- 先跑通 GPT-2/BERT/ResNet 的统一调用链

## Phase B：二阶统一到 HVP（1~2 天）

- 实现 `second-order-hvp`（框架层默认只暴露 HVP 二阶）
- 接入 full / memory-efficient 两种 HVP 模式
- 增加 hvp_batches、chunk_size、hvp_mode 配置

## Phase C：P0~P2 回归验证（1 天）

- GPT-2 Small + WikiText-103（先作为 quick sanity）
- GPT-2 Medium + WikiText-103（需要先准备对应 checkpoint）
- BERT-Large + SST-2/MNLI/STS-B
- 输出 sanity 报告与差值表

## Phase D：Adapter 扩展到 P1/P3（2~4 天）

- 新增 Pythia-410M + Alpaca adapter
- 新增 ViT-L/32 + ImageNet adapter
- 不改核心 runner，仅补适配层

---

## 9. 风险与应对

1. **HVP 成本高**：
   - 应对：多级运行配置（quick / standard / paper）
2. **大模型显存紧张**：
   - 应对：memory-efficient HVP + chunk + batch 降级
3. **跨任务指标不统一**：
   - 应对：统一保留 loss 主指标，任务指标作为附加列
4. **当前 BERT/STS-B 适配不一致导致结果无效**：
   - 应对：Phase 0 先修正模型 head 与 metric，再启动大规模对比

---

## 10. 本次修订结论

重构后将从“单一 CLI 脚本”升级为“**可复用实验框架**”，并且二阶路径统一为 **HVP-only**。这会让你可以在同一套协议下稳定比较不同模型和任务上的一阶/二阶效果，并直接对接 Table/Figure 产出链路。
