# 实验框架使用指南

> 重构完成于 2026-02-26，本文档描述新的项目结构和实验运行方式。

## 一、新增模块概览

### 1. 统一剪枝接口 `src/ckpt_compress/pruning/`

将所有方法的重要性度量和剪枝率分配统一到一个框架中，支持任意组合：

```
src/ckpt_compress/pruning/
├── __init__.py        # 导出所有公共接口
├── importance.py      # 重要性得分计算（4 种方法）
├── allocation.py      # 剪枝率分配策略（2 种策略）
└── pruner.py          # 统一剪枝器 + 工具函数
```

已注册的重要性方法：

| 名称 | 公式 | 对应论文方法 |
|------|------|-------------|
| `magnitude` | `\|w_i\|` | Magnitude baseline |
| `first-order` | `\|g_i · w_i\|` | Inshrinkerator 的核心度量 |
| `second-order` | `\|-g_i·w_i + α·h_i·w_i²\|` | **Ours**（论文方法） |
| `residual-magnitude` | `\|W_t - W_ref\|` | ExCP 的核心度量 |

已注册的分配策略：

| 名称 | 说明 |
|------|------|
| `uniform` | 所有层相同剪枝率 |
| `gamma-adaptive` | Gamma 分布拟合 + 二分法求全局阈值（论文 Eq.8-10） |

用法示例：

```python
from src.ckpt_compress.pruning import Pruner, filter_prunable_params

# 创建剪枝器（任意组合）
pruner = Pruner(importance='second-order', allocation='gamma-adaptive', alpha=0.5)

# 过滤可剪枝参数（排除 embedding/bias/layernorm）
prunable_weights = filter_prunable_params(weights)

# 计算得分 → 分配剪枝率 → 剪枝
scores = pruner.compute_scores(prunable_weights, gradients, exp_avg_sq)
layer_ratios = pruner.compute_layer_ratios(scores, global_ratio=0.5)
model, masks, actual_ratio = pruner.prune(model, scores, layer_ratios, device)
```

### 2. 实验共享库 `experiments/lib/`

消除实验脚本中的代码重复（原来 `fit_gamma_distribution` 在 9 个文件中重复定义）：

```
experiments/lib/
├── __init__.py        # 包初始化
├── models.py          # 统一模型加载（6 种模型）
├── data.py            # 统一数据加载（7 种数据集）+ 批次缓存
├── gradient.py        # 梯度和 Adam 状态收集
├── evaluation.py      # 统一评估（LM/CLS/CV 三种任务）
└── results.py         # 结果保存（CSV + JSON）
```

支持的模型：`gpt2-small`, `gpt2-medium`, `bert-base`, `bert-large`, `resnet18`, `resnet50`

支持的数据集：`wikitext103`, `wikitext2`, `sst2`, `mnli`, `stsb`, `cifar10`, `cifar100`

### 3. 统一实验脚本 `experiments/scripts/run_*.py`

每个脚本对应论文中的一个图/表：

| 脚本 | 论文位置 | 功能 |
|------|---------|------|
| `run_table1.py` | Table 1 | 固定剪枝比例下 5 种方法的质量对比 |
| `run_table3_ablation.py` | Table 3 | 消融实验（拆解二阶 + 分布放松的独立增益） |
| `run_fig4_pareto.py` | Fig 4 | Pareto 曲线（剪枝比例 vs 质量） |
| `run_fig5_heatmap.py` | Fig 5 | Sub-layer 剪枝率热力图（展示层间异质性） |
| `run_fig6_fault_tolerant.py` | Fig 6 | 容错训练模拟（多次恢复的累积误差） |

所有脚本共享相同的参数风格：

```bash
python experiments/scripts/run_xxx.py \
    --model <模型名> \
    --dataset <数据集名> \
    --checkpoint <检查点路径（可选）> \
    --device cuda \
    --output_dir results/paper_results/xxx
```

## 二、实验运行指南

### Table 1: 主实验

```bash
# GPT-2 Small 快速验证
python experiments/scripts/run_table1.py \
    --model gpt2-small --dataset wikitext103 \
    --prune_ratios 0.5,0.7,0.8,0.9 \
    --num_steps 50 --eval_batches 10 --device cuda

# GPT-2 Medium 论文实验
python experiments/scripts/run_table1.py \
    --model gpt2-medium --dataset wikitext103 \
    --prune_ratios 0.5,0.7,0.8,0.9 \
    --num_steps 100 --eval_batches 20 --device cuda

# BERT-Large on SST-2
python experiments/scripts/run_table1.py \
    --model bert-large --dataset sst2 \
    --checkpoint checkpoints/bert_large_sst2_1000steps/final.pt \
    --prune_ratios 0.5,0.7,0.8,0.9 --device cuda

# ResNet18 on CIFAR-10
python experiments/scripts/run_table1.py \
    --model resnet18 --dataset cifar10 \
    --prune_ratios 0.5,0.7,0.8,0.9 --device cuda
```

Table 1 对比的 5 种方法：
- `magnitude+uniform` — 最弱 baseline
- `first-order+uniform` — Inshrinkerator 的剪枝策略
- `residual-magnitude+uniform` — ExCP 的剪枝策略
- `second-order+uniform` — 只加二阶（消融）
- `second-order+gamma-adaptive` — **完整方法**

### Table 3: 消融实验

```bash
python experiments/scripts/run_table3_ablation.py \
    --model gpt2-small --dataset wikitext103 \
    --prune_ratios 0.3,0.5,0.7,0.9 \
    --num_steps 100 --eval_batches 20 --device cuda
```

5 种消融组合：A(mag+uni) → B(1st+uni) → C(2nd+uni) → D(1st+gamma) → E(2nd+gamma)

### Fig 4: Pareto 曲线

```bash
python experiments/scripts/run_fig4_pareto.py \
    --model gpt2-small --dataset wikitext103 \
    --min_ratio 0.1 --max_ratio 0.9 --step 0.1 \
    --num_steps 100 --eval_batches 20 --device cuda
```

输出 PNG + PDF 图片到 `results/paper_results/fig4/`。

### Fig 5: Sub-layer 剪枝率热力图

```bash
# 单方法热力图
python experiments/scripts/run_fig5_heatmap.py \
    --model gpt2-small --dataset wikitext103 \
    --global_prune_ratio 0.3 \
    --num_steps 100 --device cuda

# 多方法对比（Gamma-adaptive vs Uniform 的差异一目了然）
python experiments/scripts/run_fig5_heatmap.py \
    --model gpt2-small --dataset wikitext103 \
    --methods second-order+gamma-adaptive,first-order+gamma-adaptive \
    --global_prune_ratio 0.3 --device cuda

# 多个全局剪枝率
python experiments/scripts/run_fig5_heatmap.py \
    --model gpt2-small --dataset wikitext103 \
    --global_prune_ratios 0.3,0.5,0.7 --device cuda

# BERT-Large
python experiments/scripts/run_fig5_heatmap.py \
    --model bert-large --dataset sst2 \
    --checkpoint checkpoints/bert_large_sst2_1000steps/final.pt \
    --global_prune_ratio 0.3 --device cuda
```

输出热力图（PNG + PDF）和 CSV 数据到 `results/paper_results/fig5/`。

### Fig 6: 容错训练

```bash
python experiments/scripts/run_fig6_fault_tolerant.py \
    --model gpt2-small --dataset wikitext103 \
    --total_steps 1000 --num_recoveries 5 \
    --prune_ratio 0.5 --device cuda
```

输出 loss 曲线图到 `results/paper_results/fig6/`。

## 三、项目目录结构（重构后）

```
ckpt-compress/
├── src/ckpt_compress/
│   ├── core/base.py                    # BaseCompressor 抽象基类
│   ├── pruning/                        # 【新增】统一剪枝框架
│   │   ├── importance.py               #   重要性得分（4 种方法）
│   │   ├── allocation.py               #   剪枝率分配（2 种策略）
│   │   └── pruner.py                   #   统一剪枝器
│   ├── methods/
│   │   ├── adam_prune/                  # AdamPrune 核心实现（保留）
│   │   ├── excp/                        # ExCP 压缩方法（保留）
│   │   ├── inshrinkerator/              # Inshrinkerator 压缩方法（保留）
│   │   └── predictive/                  # PredictiveResidual（保留）
│   ├── models/                          # 模型封装（GPT-2/BERT/ResNet）
│   └── utils/                           # 工具（trainer/data_loader/tensor_ops）
│
├── experiments/
│   ├── lib/                             # 【新增】实验共享库
│   │   ├── models.py                    #   统一模型加载
│   │   ├── data.py                      #   统一数据加载 + 缓存
│   │   ├── gradient.py                  #   梯度收集
│   │   ├── evaluation.py                #   统一评估
│   │   └── results.py                   #   结果保存
│   ├── scripts/
│   │   ├── run_table1.py                # 【新增】Table 1 主实验
│   │   ├── run_table3_ablation.py       # 【新增】Table 3 消融
│   │   ├── run_fig4_pareto.py           # 【新增】Fig 4 Pareto
│   │   ├── run_fig5_heatmap.py          # 【新增】Fig 5 Sub-layer 热力图
│   │   ├── run_fig6_fault_tolerant.py   # 【新增】Fig 6 容错训练
│   │   ├── finetune/                    #   微调脚本（保留）
│   │   ├── analysis/                    #   分析脚本（保留）
│   │   └── comparison/                  #   旧对比脚本（已归档副本）
│   └── configs/                         #   实验配置
│
├── results/paper_results/               # 论文结果输出
│   ├── table1/                          #   Table 1 CSV/JSON
│   ├── table3/                          #   Table 3 CSV/JSON
│   ├── fig4/                            #   Fig 4 PNG/PDF
│   ├── fig5/                            #   Fig 5 热力图 PNG/PDF/CSV
│   └── fig6/                            #   Fig 6 PNG/PDF
│
├── archive/old_scripts/                 # 【新增】归档的旧脚本
├── checkpoints/                         # 微调检查点
├── data/                                # 数据集
├── paper/EMNLP_26.tex                   # 论文
├── plan/                                # 规划文档
│   ├── experiment_plan.md               #   实验计划
│   ├── refactoring_proposal.md          #   重构建议（诊断文档）
│   └── new_framework_guide.md           #   本文档
└── tests/                               # 测试
```

## 四、扩展指南

### 添加新的重要性方法

```python
# 在 src/ckpt_compress/pruning/importance.py 中

class MyNewScorer(ImportanceScorer):
    @property
    def name(self) -> str:
        return "my-method"
    
    def score(self, weights, gradients=None, exp_avg_sq=None, reference_weights=None):
        # 实现你的重要性计算
        return {name: ... for name, w in weights.items()}

# 注册
IMPORTANCE_REGISTRY["my-method"] = MyNewScorer
```

然后在实验脚本中直接使用：`Pruner(importance='my-method', allocation='uniform')`

### 添加新的分配策略

```python
# 在 src/ckpt_compress/pruning/allocation.py 中

class MyAllocation(AllocationStrategy):
    @property
    def name(self) -> str:
        return "my-allocation"
    
    def allocate(self, scores, global_prune_ratio):
        return {name: ... for name in scores}

ALLOCATION_REGISTRY["my-allocation"] = MyAllocation
```

### 添加新的模型

```python
# 在 experiments/lib/models.py 中

MODEL_REGISTRY['pythia-410m'] = {
    'factory': lambda **kw: get_pythia_410m(**kw),
    'type': 'lm',
    'params': '410M',
}
```

## 五、与旧代码的关系

| 旧代码 | 新代码 | 说明 |
|--------|--------|------|
| `run_main_experiment.py` | `run_table1.py` | 新增 ExCP/Inshrinkerator baseline |
| `run_ablation.py` | `run_table3_ablation.py` | 相同功能，使用统一接口 |
| `run_pareto.py` | `run_fig4_pareto.py` | 相同功能，使用统一接口 |
| `run_fault_tolerant.py` | `run_fig6_fault_tolerant.py` | 新增多种 baseline |
| 9 处重复的 `fit_gamma_distribution` | `pruning/allocation.py` | 统一到一处 |
| 各脚本中重复的 `cache_batches` 等 | `experiments/lib/` | 统一到共享库 |

旧脚本已归档到 `archive/old_scripts/`，不影响新框架运行。

## 六、下一步工作

按 experiment_plan.md 的优先级：

1. **P0**: 用 `run_table1.py` 跑 GPT-2 Medium + WikiText-103，产出 Table 1
2. **P2**: 用 `run_table3_ablation.py` 跑消融实验，产出 Table 3
3. **P3**: 用 `run_fig4_pareto.py` 跑 Pareto 曲线，产出 Fig 4
4. **P4**: 用 `run_fig6_fault_tolerant.py` 跑容错训练，产出 Fig 6
5. **P5**: 扩展到 BERT-Large + GLUE 任务
6. **P6**: 端到端压缩比对比（Table 2，需要为 AdamPrune 添加量化步骤）
