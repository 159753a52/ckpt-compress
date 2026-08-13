# DACP - Distribution-Aware Checkpoint Pruning

DACP 论文实验代码。当前论文方法使用 Taylor/HVP damage score 与 Weibull
矩估计分配，在同一 residual-recovery 协议下比较无压缩、ExCP-style、
Inshrinkerator-style 和 DACP。主表实验、方法 fidelity 与结果 provenance 的
唯一入口见 [`docs/paper_experiment_protocol.md`](docs/paper_experiment_protocol.md)。

---

## 安装

wheel 只分发可复用的 `dacp` 与 `baselines` 库代码：

```bash
pip install -e .
```

`experiments/` 是论文实验编排与证据校验层，不包含在 wheel 中。运行论文
manifest、恢复实验或生成主结果时，必须在完整源码仓库根目录执行；仅安装 wheel
不能提供 paper runner。

---

## 项目结构

```
dacp/                    # 我们的方法
  pruning/                 重要性评分 × 分配策略（可组合）
  quantization/            INT4 / DDSketch K-means 量化
  tools/                   HVP 计算工具
  models/                  GPT-2 / BERT / ResNet 封装
  utils/                   数据加载器、路径解析、训练器
baselines/               # 对比方法
  excp/                    ExCP (ICML '24)
  inshrinkerator/          Inshrinkerator (SoCC '24) + Per-type 搜索/分配
experiments/
  configs/                 论文 manifest（主实验唯一配置源）
  lib/                     统一实验框架（模型加载/评估/结果存储）
  scripts/                 统一论文 runner、诊断与历史兼容脚本
  scripts/finetune/        模型微调脚本
scripts/                 辅助验证与历史服务器脚本
```

---

## 运行实验

能力边界：

| 入口/层 | 支持范围 |
|---------|---------|
| `experiments/lib/models.py` | 统一模型注册与加载层。目前注册 `gpt2-small`、`gpt2-medium`、`gpt2-large`、`bert-base`、`bert-large`、`resnet18`、`resnet50`、`pythia-410m`、`pythia-1b`、`vit-l-32`、`vit-b-16`。这是模型工厂的注册表，不是所有脚本的共同能力声明。 |
| `experiments/scripts/run_paper_experiments.py` | 只运行 `experiments/configs/paper_experiments.yaml` 声明的五个默认 workload：`gpt2_medium_wikitext103`、`gpt2_large_wikitext103`、`bert_large_mnli`、`pythia_410m_alpaca`、`pythia_1b_alpaca`。CLI 选择仍受 manifest 中各 workload 的 ratio、K 和 seed 声明约束。 |
| `experiments/scripts/` 中的 legacy diagnostics | 每个脚本按自身 parser、数据加载器和层名解析器决定支持范围；示例中的 `gpt2-medium`、`wikitext103` 不是跨脚本保证。它们用于单 checkpoint、历史兼容或开发诊断，不能由模型注册表反推出对 `gpt2-large`、`pythia-1b` 或其他组合的支持。 |
| wheel / paper runner | wheel 只包含 `dacp` 与 `baselines` 库；`experiments/`、manifest runner 和论文结果证据链必须从完整源码仓库运行。安装 wheel 不会提供 paper runner。 |

`gpt2-large` 与 `pythia-1b` 在统一模型注册层中是已注册名称；这不表示每个 legacy diagnostic 或微调入口都支持它们。论文实验仍以 manifest 声明的 workload 为准。

先用 manifest 的 dry-run 核对实验清单：

```bash
python experiments/scripts/run_paper_experiments.py \
  --dry-run \
  --workload gpt2_medium_wikitext103 \
  --prune-ratio 0.5 \
  --recoveries 1 \
  --seeds 42
```

旧的 Python Table/Figure 脚本仅用于开发诊断和历史结果兼容，不应混入
当前论文主表。绑定个人 Conda、目录和旧方法名的服务器 launcher 已删除；
GPU 机器也应直接使用上面的 manifest runner，并通过参数或环境变量映射路径。
复杂度与耗时记录边界见
[`docs/complexity_analysis.md`](docs/complexity_analysis.md)；未经目标机器 JSON
实测的小时数不作为论文证据。
`experiments/scripts/aggregate_results.py` 只用于读取历史结果 schema；论文
主结果必须从 digest-verified suite JSON 聚合，不能把该兼容扫描器的输出当作
完整性证明。

### 诊断脚本

```bash
# 旧方法组合诊断
python experiments/scripts/run_method_comparison.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 联合压缩诊断
python experiments/scripts/run_joint_compression.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 消融诊断
python experiments/scripts/run_ablation_study.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# Pareto 诊断
python experiments/scripts/run_pareto_curves.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 剪枝率热力图
python experiments/scripts/run_pruning_heatmap.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 旧容错训练诊断
python experiments/scripts/run_fault_tolerant_training.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 分布拟合诊断
python experiments/scripts/run_gamma_validation.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 敏感性与计时诊断
python experiments/scripts/run_sensitivity_analysis.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 层分布 Violin 图
python experiments/scripts/run_layer_distribution_violin.py \
  --model gpt2-medium --dataset wikitext103 --device cuda
```

### Makefile 开发入口

```bash
make test          # 全量测试
make test-cov      # 全 dacp 包覆盖率报告
make test-paper-cov # 论文核心选择模块 90% 覆盖率门禁
make typecheck-paper # 论文 runner、证据链和模型/数据边界类型检查
make lint          # 当前论文路径的格式与编译检查
make paper-plan    # 仅显示统一论文实验清单
make wheel         # 构建 wheel
```

### 模型与脚本边界

需要 `--model` 的 legacy diagnostic 只接受其自身实现和数据路径覆盖的组合；请先运行对应入口的 `--help`，再按该脚本的示例准备 checkpoint 与数据。微调入口通常固定模型族或任务，不能仅因为模型出现在注册表中就替换为任意名称。统一论文实验入口应使用上面的 manifest workload 名称，而不是把模型名直接当作 workload。

---

## 旧诊断框架的方法组合

| 重要性评分 | 分配策略 | 说明 |
|-----------|---------|------|
| `magnitude` | `uniform` | 幅度剪枝（基线） |
| `first-order` | `uniform` | 一阶梯度 |
| `first-order` | `gamma-adaptive` | 一阶 + 自适应分配 |
| `residual-magnitude` | `uniform` | 残差幅度（ExCP 风格） |
| `second-order-hvp` | `uniform` | 二阶 HVP（消融用） |
| `second-order-hvp` | `gamma-adaptive` | 旧 Gamma allocation 诊断，不是当前 paper runner 的 DACP contract |

当前论文 runner 的 DACP contract 是 `taylor_weibull_mom`，见 manifest、
[`experiments/lib/paper_baselines.py`](experiments/lib/paper_baselines.py) 和实验协议。

---

## 许可证

MIT License
