# CLAUDE.md

本文件为 Claude Code 提供在此代码库中工作的快速参考指导。

## 项目概述

**DACP** (Distribution-Aware Checkpoint Pruning) — EMNLP '26 论文实验代码。

对比 DACP（二阶 HVP damage score + Gamma 自适应分层分配）与基线方法在检查点压缩任务上的表现。

**核心压缩方法**:
- **DACP (本文)**: 分布感知检查点剪枝，支持多种重要性评分（magnitude / first-order / second-order-hvp）+ Gamma 自适应分层分配
- **ExCP** (ICML '24): 残差编码 + 权重 - 动量联合剪枝 + K-means 量化
- **Inshrinkerator** (SoCC '24): 三向分区 + DDSketch 近似 K-means + RLE 增量编码 + Per-type 剪枝率搜索

**技术栈**: PyTorch, NumPy

## 快速开始

```bash
pip install -e .

# 方法对比（Table 1）
python experiments/scripts/run_method_comparison.py \
  --model gpt2-medium --dataset wikitext103 --device cuda

# 消融实验（Table 3）
python experiments/scripts/run_ablation_study.py \
  --model gpt2-medium --dataset wikitext103 --device cuda
```

## 代码库结构

```
dacp/                         # 我们的方法（Python package）
  pruning/                      核心剪枝逻辑
    importance.py                 3个重要性评分器（magnitude/first-order/residual-magnitude）
    allocation.py                 3个分配策略（uniform/gamma-adaptive/global-topk）
    pruner.py                     Pruner 统一入口
    param_schema.py               层类型推断
  quantization/                 量化模块
    int4.py                       INT4 对称量化
    kmeans.py                     DDSketch + 近似 K-means 量化
  tools/                        HVP 工具
    importance.py                 HVP 计算（compute_hvp, compute_hvp_batched, compute_importance_scores_hvp_*）
  models/                       模型封装（GPT-2/BERT/ResNet）
  utils/                        数据加载器、训练器

baselines/                    # 对比方法
  excp/                         ExCP (ICML '24) — α=5e-5, β=2.0
  inshrinkerator/               Inshrinkerator (SoCC '24)
    inshrinkerator.py             核心管线
    per_type_search.py            Per-layer-type 剪枝率搜索
    per_type_allocation.py        Per-type 分配策略
    approx_kmeans.py / sketch.py  DDSketch + 近似 K-means
    delta_encoding.py / partition.py / metrics.py

experiments/
  lib/                          统一实验框架
    models.py                     模型加载
    data.py                       数据加载
    evaluation.py                 评估
    results.py                    结果存储
    importance_compare/scoring.py 多方法得分计算
  scripts/                      论文实验脚本（10个）
    run_method_comparison.py      Table 1: 方法对比
    run_joint_compression.py      Table 2: 联合压缩
    run_ablation_study.py         Table 3: 消融实验
    run_pareto_curves.py          Fig 4: Pareto 曲线
    run_pruning_heatmap.py        Fig 5: 剪枝率热力图
    run_fault_tolerant_training.py Fig 6: 容错训练
    run_gamma_validation.py       §9.1-9.2: Gamma 验证 + 多分布对比
    run_sensitivity_analysis.py   §9.3: 敏感性分析
    run_layer_distribution_violin.py §9.4: 层分布 Violin
    run_inshrinkerator_like_search.py Per-type ratio 搜索
  scripts/finetune/             模型微调脚本（7个）

scripts/                      工具脚本
  download_data.py              数据集下载
  download_models.py            模型下载
```

## 实验命令

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

## 方法组合

| 重要性评分 | 分配策略 | 说明 |
|-----------|---------|------|
| `magnitude` | `uniform` | 幅度剪枝（基线） |
| `first-order` | `uniform` | 一阶梯度 |
| `first-order` | `gamma-adaptive` | 一阶 + 自适应分配 |
| `residual-magnitude` | `uniform` | 残差幅度（ExCP 风格） |
| `second-order-hvp` | `uniform` | 二阶 HVP（消融用） |
| **`second-order-hvp`** | **`gamma-adaptive`** | **完整方法（DACP）** |

## 数据和模型

```bash
python scripts/download_data.py --all
python scripts/download_models.py --all
```
