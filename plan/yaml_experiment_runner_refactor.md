# YAML 驱动实验运行：重构建议（替代长 CLI 参数）

## 1. 你想要的效果是什么

你现在不想用类似下面这种“长 CLI 参数串”的方式跑实验：

```bash
python experiments/scripts/run_table3_ablation.py --model ... --dataset ... --prune_ratios ...
python experiments/scripts/run_table1.py --model ... --dataset ... --prune_ratios ...
python experiments/scripts/run_fig5_heatmap.py --model ... --dataset ... --methods ...
```

更希望用 **YAML 配置文件**描述实验（模型/数据集/剪枝率/方法组合/输出目录等），然后用 Python API（或一个极薄的入口）批量执行。

> 注意：完全“零 CLI”仍然需要一个“启动方式”（比如在 notebook / 主控脚本里调用 `run_yaml()`）。这里建议把 CLI 降到只剩一个参数：`config.yaml`，其余都写在 YAML 里。

---

## 2. 当前代码现状（对齐）

你现在已经有两类东西：

1) **实验共享库**：`experiments/lib/`（models/data/evaluation/results 等）

2) **一阶/二阶(HVP)框架雏形**：`experiments/lib/importance_compare/`
- `config.py`：`ExperimentConfig`
- `runner.py`：`ImportanceCompareRunner`
- `scoring.py`/`hvp.py`：HVP-only 二阶得分与公平 batch 对齐逻辑

3) **论文脚本**（仍然是 CLI 形式）：`experiments/scripts/run_table1.py`、`run_table3_ablation.py`、`run_fig5_heatmap.py` 等  
这些脚本目前已经复用了新的 scoring/HVP 逻辑，但仍需要用 `argparse` 传参。

---

## 3. 最小侵入式重构目标

### 目标 A：用 YAML 描述实验
- 一个 YAML 文件可以包含多个 run（Table1/Table3/Fig5 等）。
- 支持 `defaults`（默认参数）+ per-run override。
- 支持“方法组合”（importance + allocation）列表。

### 目标 B：所有脚本逻辑下沉到可复用的 runner
- scripts 只保留“薄 wrapper”（可选）。
- notebook/主控脚本直接 import runner + YAML config 执行。

### 目标 C：输出形式统一且可复现
- 统一输出目录结构：`results/<group>/<run_name>/<timestamp>.*`
- 每个 run 输出 `CSV + JSON`（和目前 `save_results` 一致），Fig 类 run 额外输出 `png/pdf`。

---

## 4. 建议新增的 YAML Schema（核心）

建议一个 YAML 文件描述 **多个 runs**：

```yaml
version: 1

defaults:
  device: cuda
  seed: 42
  batch_size: 4
  seq_length: 512
  num_steps: 50
  hvp_batches: 4
  hvp_mode: full      # full | block
  chunk_size: 10
  eval_batches: 20
  alpha: 0.5
  output_root: results/paper_results/yaml_runs

runs:
  - name: table3_gpt2_small_wt103
    kind: paper_table3_ablation
    model: gpt2-small
    dataset: wikitext103
    checkpoint: checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt
    prune_ratios: [0.3, 0.5, 0.7, 0.9]
    methods:
      - {label: A, importance: magnitude, allocation: uniform}
      - {label: B, importance: first-order, allocation: uniform}
      - {label: C, importance: second-order-hvp, allocation: uniform}
      - {label: D, importance: first-order, allocation: gamma-adaptive}
      - {label: E, importance: second-order-hvp, allocation: gamma-adaptive}

  - name: table1_bert_large_stsb
    kind: paper_table1
    model: bert-large
    dataset: stsb
    checkpoint: checkpoints/bert_large_stsb_1000steps/checkpoint_step_1000_final.pt
    prune_ratios: [0.5, 0.7, 0.8, 0.9]
    methods:
      - {importance: magnitude, allocation: uniform}
      - {importance: first-order, allocation: uniform}
      - {importance: residual-magnitude, allocation: uniform}
      - {importance: second-order-hvp, allocation: uniform}
      - {importance: second-order-hvp, allocation: gamma-adaptive}

  - name: fig5_gpt2_small_heatmap
    kind: fig5_heatmap
    model: gpt2-small
    dataset: wikitext103
    checkpoint: checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt
    global_prune_ratios: [0.3]
    methods:
      - second-order-hvp+gamma-adaptive
      - first-order+gamma-adaptive
```

### 字段映射（从你现在的 CLI 来）

以 `run_table3_ablation.py` 为例：
- `--model` → `model`
- `--dataset` → `dataset`
- `--checkpoint` → `checkpoint`
- `--prune_ratios "0.3,0.5,..."` → `prune_ratios: [0.3, 0.5, ...]`
- `--num_steps` → `num_steps`
- `--hvp_batches` → `hvp_batches`
- `--hvp_mode` → `hvp_mode`
- `--chunk_size` → `chunk_size`
- `--eval_batches` → `eval_batches`
- `--device` → `device`

---

## 5. 代码重构建议（关键：把脚本变成 runner）

### 5.1 抽象一个“方法规格”数据结构（所有 runner 共用）

新增：
- `experiments/lib/paper/method_spec.py`

```python
@dataclass
class MethodSpec:
    name: str                 # e.g., "E: second-order-hvp+gamma-adaptive"
    label: Optional[str]      # Table3 用
    importance: str           # magnitude / first-order / second-order-hvp / residual-magnitude
    allocation: str           # uniform / gamma-adaptive
```

用途：
- Table1/Table3：按 `MethodSpec` 列表跑
- Fig5：`methods` 字符串列表也可以先 parse 成 `MethodSpec`

### 5.2 抽象一个“通用剪枝评估 runner”（Table1/Table3 共享）

新增：
- `experiments/lib/paper/prune_eval_runner.py`

核心接口：
- 输入：`model/dataset/checkpoint/prune_ratios/method_specs/hvp_config/eval_config`
- 行为：  
  1) `cache_batches()` 固定 train/eval batch  
  2) `compute_scores_by_method()` 只按 “importance 去重” 计算 score（含 HVP）  
  3) 每个 method 用对应 allocation 计算 layer_ratios，然后 `apply_pruning()`  
  4) `evaluate()` 得到 loss/acc/ppl/pearson  
  5) `save_results()` 输出 CSV+JSON

这样 `run_table1.py` / `run_table3_ablation.py` 就只需要：
- `cfg = load_yaml(...); runner.run(cfg)`

### 5.3 Fig5：把“计算 layer_ratios + 聚合 + 画图”提成 lib

建议把 `experiments/scripts/run_fig5_heatmap.py` 的核心逻辑拆到：
- `experiments/lib/fig5/heatmap_runner.py`
- `experiments/lib/fig5/parsing.py`（parse_gpt2_param/parse_bert_param 等）
- `experiments/lib/fig5/plotting.py`（matplotlib/seaborn 画图）

原因：
- YAML runner 调用时就不需要 `argparse` 和脚本级 main。
- Fig5 的方法列表也自然来自 YAML。

---

## 6. YAML Runner：统一入口（推荐实现）

新增：
- `experiments/lib/yaml_runner.py`

提供两个入口（满足“非 CLI 中心”）：

1) **Python API（推荐）**
```python
from experiments.lib.yaml_runner import run_yaml
run_yaml("experiments/configs/paper/paper_runs.yaml")
```

2) **可选薄 CLI（仅 1 个参数）**
```bash
python experiments/scripts/run_yaml.py --config experiments/configs/paper/paper_runs.yaml
```

`run_yaml()` 做的事：
- `yaml.safe_load()`
- 合并 `defaults` 与 `run` override
- 按 `kind` 分发到对应 runner（registry）
- 每个 run 统一写入：`output_root/<name>/...`

建议加一个最小 registry：
- `"paper_table1"` → `PruneEvalRunner`
- `"paper_table3_ablation"` → `PruneEvalRunner`（只是 methods 不同）
- `"fig5_heatmap"` → `Fig5HeatmapRunner`

---

## 7. 输出结果建议（保持与现有一致，但更结构化）

建议目录结构：

```
results/paper_results/yaml_runs/
  table3_gpt2_small_wt103/
    table3_gpt2_small_wt103_YYYYmmdd_HHMMSS.csv
    table3_gpt2_small_wt103_YYYYmmdd_HHMMSS.json
  table1_bert_large_stsb/
    table1_bert_large_stsb_YYYYmmdd_HHMMSS.csv
    table1_bert_large_stsb_YYYYmmdd_HHMMSS.json
  fig5_gpt2_small_heatmap/
    fig5_*.csv / *.json
    fig5_*.png / *.pdf
```

字段形式：
- 通用：`method/importance/allocation/target_ratio/actual_ratio`
- 指标：  
  - LM：`loss/perplexity`  
  - CLS：`loss/accuracy`  
  - REG(STS-B)：`loss/pearson`
- baseline：`baseline_loss/baseline_accuracy/baseline_perplexity/...`

---

## 8. 风险点与建议（确保 YAML 运行的结果可信）

1) **公平性协议不可退化**  
必须确保：
- score 估计 batch 与评估 batch 是固定缓存（`cache_batches`）
- HVP/梯度都在同一批次集合上平均（不要 “grad 用 1 个 batch，HVP 用多个 batch”）

2) **BERT/GLUE head 必须匹配 checkpoint**  
YAML 中 `dataset: stsb` 时应自动走 `reg`（num_labels=1）并输出 pearson。

3) **Fig6 容错训练目前不是 HVP-only**  
如果你希望 Fig6 也 YAML 化，建议先明确：
- 是否接受 Adam 近似二阶（快，但与 HVP-only 主张不完全一致）
- 若强制 HVP-only，运行成本会显著上升（每次 recovery 都要多次反传）。

---

## 9. 最终建议（落地顺序）

1) 先实现 `yaml_runner.py` + `PruneEvalRunner`（覆盖 Table1/Table3 需求）
2) 再把 Fig5 逻辑拆到 `experiments/lib/fig5/*`，让 YAML runner 能调用
3) 最后保留 `run_table1.py/run_table3_ablation.py/run_fig5_heatmap.py` 作为薄 wrapper（可选），确保旧用法不立刻断掉

