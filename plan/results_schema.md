# results_schema.md（结果文件格式约定）

本文件定义 ckpt-compress 各实验脚本/runner 产出的**结果文件形式**（CSV / JSON / 图片）以及**字段含义**（表格每行/每列代表什么、图的横纵坐标是什么）。

后续任何关于“结果长什么样 / 需要新增字段 / 字段含义调整 / 图表坐标调整”的讨论，都应同步更新本文件，保证它是单一可信来源（single source of truth）。

最后更新：2026-02-28

---

## 1. 通用约定（适用于所有实验）

### 1.1 输出文件的基本形态

大多数实验会写入：
- 一个 `*.csv`（长表 / tidy data）
- 一个 `*.json`（同一份结果 + 配置的结构化封装）

部分图类实验还会额外写入：
- `*.png` + `*.pdf`（可直接用于论文或报告）

CSV/JSON 的落盘由 `experiments/lib/results.py` 的 `save_results()` 完成，文件名包含时间戳（精确到秒）。

### 1.2 JSON 的顶层结构（统一）

JSON 文件结构：
- `experiment`: 实验名字符串（由脚本传入，比如 `table1_gpt2-small_wikitext103`）
- `timestamp`: 写入时间戳（YYYYmmdd_HHMMSS）
- `config`: 配置字典（通常是 argparse 的 `vars(args)` 或 runner 的 config）
- `results`: 结果行列表（每一行与 CSV 的一行字段一致）

### 1.3 任务类型与指标键

按 `task_type` 不同，指标列不同：
- `lm`（语言模型）：`loss`、`perplexity`
- `cls`（分类）：`loss`、`accuracy`
- `reg`（回归，STS-B）：`loss`、`pearson`
- `cv`（图像分类）：`loss`、`accuracy`

### 1.4 剪枝率字段

- `target_ratio`：目标全局剪枝率（0~1，小数；0.7 表示剪掉 70%）
- `actual_ratio`：实际全局剪枝率（0~1，小数；由于阈值/离散性可能与 target 有微小偏差）

### 1.5 方法标识字段

统一用两维描述方法：
- `importance`：重要性度量（例如 `magnitude` / `first-order` / `residual-magnitude` / `second-order-hvp`）
- `allocation`：剪枝率分配策略（例如 `uniform` / `gamma-adaptive`）

多数脚本还会提供：
- `method`：拼接后的字符串（例如 `second-order-hvp+gamma-adaptive`，或 Table3 中带 label 的版本）

### 1.6 baseline 对照字段（如有）

多数表类/对比类实验会在每行携带 baseline 信息：
- `baseline_loss`
- `baseline_perplexity` / `baseline_accuracy` / `baseline_pearson`（按任务类型）

并派生：
- `loss_increase` = `loss - baseline_loss`
- `loss_increase_pct` = `loss_increase / baseline_loss * 100`

---

## 2. Table 1（主表）：固定剪枝率下的质量对比

脚本：`experiments/scripts/run_table1.py`

### 2.1 结果文件

默认输出目录：`results/paper_results/table1/`
- `table1_<model>_<dataset>_<timestamp>.csv`
- `table1_<model>_<dataset>_<timestamp>.json`

### 2.2 CSV：每一行代表什么

一行 = “某个方法（importance+allocation）在某个 `target_ratio` 下剪枝一次，并在固定的验证缓存 batch 上评估得到的指标”。

### 2.3 CSV：关键列含义（通用列）

| 列名 | 含义 |
|---|---|
| `method` | 方法名（`importance+allocation`） |
| `importance` | 重要性度量名称 |
| `allocation` | 分配策略名称 |
| `target_ratio` | 目标剪枝率 |
| `actual_ratio` | 实际剪枝率 |
| `loss` | 剪枝后评估 loss |
| `baseline_loss` | 未剪枝 baseline 的评估 loss |
| `loss_increase` | `loss - baseline_loss` |
| `loss_increase_pct` | `loss_increase / baseline_loss * 100` |

### 2.4 CSV：按任务类型出现的列

- `task_type = lm`：
  - `perplexity`：剪枝后困惑度（由 `exp(loss)` 得到）
  - `baseline_perplexity`：baseline 困惑度

- `task_type = cls/cv`：
  - `accuracy`：剪枝后准确率
  - `baseline_accuracy`：baseline 准确率
  - `accuracy_drop`（如存在）：`baseline_accuracy - accuracy`

- `task_type = reg`：
  - `pearson`：剪枝后 Pearson 相关系数
  - `baseline_pearson`：baseline Pearson

### 2.5 论文表格的行/列如何对应

论文展示一般做成“宽表”：
- 行（rows）：方法（如 5 行：Magnitude+Uniform、First-order+Uniform、Residual-mag+Uniform、Second-order-hvp+Uniform、Second-order-hvp+Gamma-adaptive）
- 列（cols）：剪枝率（50% / 70% / 80% / 90%）
- 单元格：质量指标（LM 用 ppl↓；分类/CV 用 acc↑；回归用 pearson↑；可另附 loss↑）

---

## 3. Table 3（消融）：A~E 五组合

脚本：`experiments/scripts/run_table3_ablation.py`

### 3.1 结果文件

默认输出目录：`results/paper_results/table3/`
- `table3_<model>_<dataset>_<timestamp>.csv`
- `table3_<model>_<dataset>_<timestamp>.json`

### 3.2 CSV：每一行代表什么

一行 = “某个消融组合（A/B/C/D/E）在某个 `target_ratio` 下剪枝一次并评估得到的指标”。

### 3.3 CSV：关键列含义

| 列名 | 含义 |
|---|---|
| `label` | 消融标识（A~E） |
| `importance` | 重要性度量（A/B/C/D/E 对应不同设置） |
| `allocation` | 分配策略（uniform / gamma-adaptive） |
| `method` | 带 label 的方法描述字符串 |
| `target_ratio` | 目标剪枝率 |
| `actual_ratio` | 实际剪枝率 |
| 指标列 | 与 Table1 同（按 task_type 变化） |

### 3.4 论文表格的行/列如何对应

论文展示一般做成“宽表”：
- 行（rows）：A~E
- 列（cols）：剪枝率（例如 50% / 70% / 90%）
- 单元格：质量指标（ppl/acc/pearson）或 loss_increase

---

## 4. Fig 4（Pareto 曲线）：扫描剪枝率画曲线

脚本：`experiments/scripts/run_fig4_pareto.py`

### 4.1 结果文件

默认输出目录：`results/paper_results/fig4/`
- `fig4_<model>_<dataset>_<timestamp>.csv`
- `fig4_<model>_<dataset>_<timestamp>.json`
- `fig4_<model>_<dataset>.png`
- `fig4_<model>_<dataset>.pdf`

### 4.2 曲线图的横纵坐标

- X 轴：`Pruning Ratio`（剪枝率；脚本扫描 `min_ratio ~ max_ratio`）
- Y 轴：质量指标（按 task_type）
  - `lm`：`Perplexity`（通常使用 log y 轴）
  - `cls/cv`：`Accuracy`
  - `reg`：`Pearson`

### 4.3 CSV：每一行代表什么

一行 = “某个方法在某个 `target_ratio` 下的评估点”，用于画曲线上的点。

常见列：
- `method/importance/allocation/target_ratio/actual_ratio`
- `loss` +（`perplexity` 或 `accuracy` 或 `pearson`）
- `baseline_*`（用于对照）

---

## 5. Fig 5（热力图）：层/子层剪枝率分配

脚本：`experiments/scripts/run_fig5_heatmap.py`

### 5.1 结果文件

默认输出目录：`results/paper_results/fig5/`

脚本会产生两类“数据文件”和两类“图文件”：

1) 全局汇总（长表）：
- `fig5_<model>_<dataset>_<timestamp>.csv`
- `fig5_<model>_<dataset>_<timestamp>.json`

2) 单方法/单剪枝率的矩阵 CSV（百分比矩阵）：
- `fig5_rates_<safe_method>_<ratio_str>.csv`（例如 `fig5_rates_second-order-hvp_gamma-adaptive_30pct.csv`）

3) 单方法热力图：
- `fig5_heatmap_<safe_method>_<ratio_str>.png`
- `fig5_heatmap_<safe_method>_<ratio_str>.pdf`

4) 多方法对比热力图（并排）：
- `fig5_comparison_<ratio_str>.png`
- `fig5_comparison_<ratio_str>.pdf`

其中：
- `<safe_method>` 是把方法名中的 `+` 替换为 `_` 的版本
- `<ratio_str>` 例如 `30pct`

### 5.2 热力图的横纵坐标与颜色含义

- X 轴：`Parameter Group`（参数组类型，如 GPT-2 的 attn_qkv/mlp_fc/...）
- Y 轴：`Block`（Transformer block 编号）
- 颜色：剪枝率（脚本以百分比显示）

### 5.3 汇总 CSV（fig5_*.csv）：每一行代表什么

一行 = “某个方法在某个 `global_ratio` 下，对某个 block 的某个 param_group 分配的剪枝率”。

| 列名 | 含义 |
|---|---|
| `method` | 方法名（`importance+allocation`） |
| `global_ratio` | 目标全局剪枝率（与 Table 的 target_ratio 同义，但这里命名为 global_ratio） |
| `block_id` | block 编号（int） |
| `param_group` | 参数组名称（字符串） |
| `prune_rate` | 剪枝率（0~1） |
| `prune_rate_pct` | 剪枝率百分比（0~100） |

### 5.4 单方法矩阵 CSV（fig5_rates_*.csv）：每个单元格代表什么

- 行索引：`block_id`
- 列：`param_group`
- 单元格值：该 block / param_group 的剪枝率（百分比）

---

## 6. Fig 6（容错训练模拟）：多次恢复的累积误差

脚本：`experiments/scripts/run_fig6_fault_tolerant.py`

### 6.1 结果文件

默认输出目录：`results/paper_results/fig6/`
- `fig6_<model>_<dataset>_<timestamp>.csv`
- `fig6_<model>_<dataset>_<timestamp>.json`
- `fig6_<model>_<dataset>.png`
- `fig6_<model>_<dataset>.pdf`

### 6.2 曲线图的横纵坐标

- X 轴：`Training Steps`（训练步数）
- Y 轴：`Validation Loss`（验证 loss）

图中竖虚线：每次“恢复点”的位置（segment 边界）。

### 6.3 CSV：每一行代表什么

一行 = “某个方法在某个训练步数 step 处的验证指标记录点（按 eval_interval 采样）”。

| 列名 | 含义 |
|---|---|
| `method` | 方法名（例如 `none`、`magnitude+uniform` 等） |
| `step` | 训练步数（int） |
| `loss` | 验证 loss |
| `perplexity` | 若为 LM 则为 ppl，否则通常为 0 |
| `accuracy` | 若为分类/CV 则为 acc，否则通常为 0 |

备注：Fig6 当前二阶路径为 Adam 近似（不是 HVP-only）。

---

## 7. 变更记录（Changelog）

- 2026-02-28：初始化版本，覆盖 Table1/Table3/Fig4/Fig5/Fig6 的输出文件与字段语义。

