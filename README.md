# DACP — Distribution-Aware Checkpoint Pruning

EMNLP '26 实验代码。对比 DACP（二阶 HVP + Gamma 自适应分配）与基线方法在检查点压缩任务上的表现。

---

## 安装

```bash
pip install -e .
```

---

## 项目结构

```
dacp/                    # 我们的方法
  pruning/                 重要性评分 × 分配策略（可组合）
  quantization/            INT4 / DDSketch K-means 量化
  tools/                   HVP 计算工具
  models/                  GPT-2 / BERT / ResNet 封装
  utils/                   数据加载器、训练器
baselines/               # 对比方法
  excp/                    ExCP (ICML '24)
  inshrinkerator/          Inshrinkerator (SoCC '24) + Per-type 搜索/分配
experiments/
  lib/                     统一实验框架（模型加载/评估/结果存储）
  scripts/                 论文各表/图对应的实验脚本（10个）
  scripts/finetune/        模型微调脚本（7个）
scripts/                 数据集和模型下载脚本
```

---

## 运行实验

### 单个实验

```bash
# 方法对比（论文 Table 1）
python experiments/scripts/run_method_comparison.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 联合压缩（Table 2）
python experiments/scripts/run_joint_compression.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 消融实验（Table 3）
python experiments/scripts/run_ablation_study.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# Pareto 曲线（Figure 4）
python experiments/scripts/run_pareto_curves.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 剪枝率热力图（Figure 5）
python experiments/scripts/run_pruning_heatmap.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 容错训练（Figure 6）
python experiments/scripts/run_fault_tolerant_training.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# Gamma 分布验证（§9.1-9.2）
python experiments/scripts/run_gamma_validation.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 敏感性分析 + 计时分解（§9.3）
python experiments/scripts/run_sensitivity_analysis.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 层分布 Violin 图（§9.4）
python experiments/scripts/run_layer_distribution_violin.py \
  --model gpt2-medium --dataset wikitext103 --device cuda
```

### Makefile 快捷方式

```bash
make table1        # 方法对比
make table2        # 联合压缩
make table3        # 消融实验
make fig4          # Pareto 曲线
make fig5          # 剪枝率热力图
make fig6          # 容错训练
make gamma         # Gamma 验证
make sensitivity   # 敏感性分析
make violin        # 层分布 Violin
make all           # 全部实验
```

### 切换模型

所有脚本支持 `--model` 参数：`gpt2-small` / `gpt2-medium` / `bert-base` / `bert-large` / `resnet18` / `resnet50` / `pythia-410m` / `vit-l-32` / `vit-b-16`

---

## 支持的方法组合

| 重要性评分 | 分配策略 | 说明 |
|-----------|---------|------|
| `magnitude` | `uniform` | 幅度剪枝（基线） |
| `first-order` | `uniform` | 一阶梯度 |
| `first-order` | `gamma-adaptive` | 一阶 + 自适应分配 |
| `residual-magnitude` | `uniform` | 残差幅度（ExCP 风格） |
| `second-order-hvp` | `uniform` | 二阶 HVP（消融用） |
| **`second-order-hvp`** | **`gamma-adaptive`** | **完整方法（DACP）** |

---

## 许可证

MIT License
