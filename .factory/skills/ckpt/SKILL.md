---
name: ckpt
description: >
  Research assistant for checkpoint compression paper (EMNLP 2026). Use when the user needs to:
  (1) design or run experiments comparing first-order vs second-order importance pruning,
  (2) implement or debug distribution-aware (Gamma-based) adaptive layer-wise pruning rate allocation,
  (3) validate that second-order information yields better pruning accuracy/loss than first-order,
  (4) validate that distributional relaxation yields better results than uniform pruning rates,
  (5) write or modify experiment scripts, analyze results, generate figures/tables for the paper,
  (6) fine-tune models (GPT-2, BERT) and run pruning experiments on NLP/CV workloads,
  (7) any task related to the ckpt-compress project or the EMNLP 2026 paper.
---

# 检查点压缩研究 Skill

## 语言规则

与用户对话时必须使用中文。代码注释和变量名可以用英文，但所有解释、分析、建议、总结都用中文。

## Skill 备份规则

一旦修改这个 skill，在 `/lihongliang/fangzl/skill` 目录下备份一份，复制过去即可：
```bash
cp -r /lihongliang/fangzl/ckpt-compress/.factory/skills/ckpt /lihongliang/fangzl/skill/
```

## 第一步 — 每次都要做

先阅读论文，了解当前研究状态：

```
Read /lihongliang/fangzl/ckpt-compress/paper/EMNLP_26.tex
```

然后阅读代码库指南：`references/codebase-guide.md`

## 研究目标

需要通过实验验证两个核心主张：

1. **二阶 > 一阶**：使用梯度和 Hessian 对角线的 damage score（`|−g_i·θ_i + 0.5·h_i·θ_i²|`）比仅用一阶（`|g_i·θ_i|`）能做出更好的剪枝决策。"更好" = 在相同稀疏度下，剪枝后的 loss 增量更小或准确率保持更高。

2. **分布放松 > 均匀剪枝**：对每层 damage score 拟合 Gamma 分布，通过二分法求解全局阈值 `c*`，得到的逐层剪枝率分配优于对所有层施加统一剪枝率。"更好" = 同上。

## 关键公式

每个参数的 damage score：
```
s_{ℓ,i} = |−g_{ℓ,i}·θ_{ℓ,i} + 0.5·h_{ℓ,i}·θ_{ℓ,i}²|
```

一阶 baseline：
```
s_{ℓ,i}^{(1)} = |g_{ℓ,i}·θ_{ℓ,i}|
```

Gamma 拟合（矩估计）：
```
k̂_ℓ = s̄_ℓ² / σ_ℓ²,  θ̂_ℓ = σ_ℓ² / s̄_ℓ
```

全局阈值方程（二分法目标）：
```
Σ_ℓ n_ℓ · γ(k̂_ℓ, c/θ̂_ℓ) / Γ(k̂_ℓ) = P · Σ_ℓ n_ℓ
```

## 实验流程

详细实验设计和执行步骤见 `references/experiment-guide.md`。

总体流程：

1. **准备**：下载模型 + 数据集，微调得到带优化器状态的检查点
2. **计算得分**：收集梯度和 Hessian 对角线，计算重要性得分
3. **剪枝 + 评估**：在多个稀疏率下执行剪枝，测量 loss/准确率
4. **对比**：一阶 vs 二阶，均匀 vs Gamma 自适应分配
5. **可视化**：生成论文所需的图表

## Hessian 对角线计算

两种方式：

- **Adam v_t 近似**：用 Adam 的 `exp_avg_sq` 作为 Hessian 对角线近似。快但不精确。
- **HVP（Hessian-Vector Product）**：通过 `torch.autograd.functional.hvp` 计算精确对角线。精确但需要 2 倍反向传播开销。

论文实验优先用 HVP 以支撑准确性声明。快速迭代时可用 Adam 近似。

相关代码：`src/ckpt_compress/methods/adam_prune/importance.py`

## 关键规则

### 磁盘空间

下载模型或数据集时，如果出现磁盘空间检查错误，**禁用磁盘空间检查**。服务器存储充足。HuggingFace 下载时设置：
```python
import os
os.environ["HF_HUB_DISABLE_DISK_SPACE_CHECK"] = "1"
```

### 文件写入

如果写入大文件时遇到错误：
1. 先创建空文件
2. 逐次追加写入，每次 ≤200 行
3. 后续写入使用追加模式

### 子代理委托

对于复杂的、上下文占用大的、可以独立运行的任务（如微调模型、运行完整实验套件、生成所有图表），委托给子代理（Task 工具）执行，保持主对话上下文干净。

### 内存管理

- 用 `--max_samples` 限制数据集大小以加速实验
- GPT-2 Medium 及以上：使用 `MemoryEfficientEvaluator`（避免 deepcopy）
- 用 `cache_batches()` 确保梯度计算和 loss 评估使用相同数据
- 监控 GPU 显存；实验间使用 `torch.cuda.empty_cache()`

### 实验公正性

- 计算重要性得分和评估 loss 必须使用**相同的数据批次**
- 对比方法时确保**完全相同的实验条件**（相同检查点、相同数据、相同稀疏率）
- 所有结果带时间戳保存以确保可复现
- NLP 任务报告 loss 和困惑度；CV 任务报告准确率

## 项目结构

```
/lihongliang/fangzl/ckpt-compress/
├── paper/EMNLP_26.tex              # 论文
├── src/ckpt_compress/
│   ├── methods/adam_prune/          # 核心剪枝实现
│   │   ├── importance.py            # 重要性得分计算
│   │   ├── distribution.py          # 分布拟合
│   │   ├── distribution_fast.py     # 快速分布拟合
│   │   ├── adaptive_pruning.py      # 自适应剪枝逻辑
│   │   └── layer_pruning.py         # 逐层剪枝
│   ├── models/                      # 模型封装（GPT-2, BERT, ResNet）
│   └── utils/                       # 数据加载器、训练器、张量操作
├── experiments/scripts/             # 实验脚本
├── checkpoints/                     # 保存的检查点
├── results/                         # 实验结果
└── data/                            # 数据集
```

## 让二阶胜出的调参技巧

如果二阶结果没有明显优于一阶，考虑：

1. **调整 alpha**：`s_i = |−g_i·θ_i + α·h_i·θ_i²|` 中的 α 很关键。尝试 α ∈ {0.1, 0.3, 0.5, 1.0, 2.0}，论文默认 0.5。
2. **更高稀疏率**：二阶优势在激进剪枝（30%–70%）时更明显。低稀疏率（5–10%）时两种方法都只移除无关紧要的参数。
3. **更多梯度累积步数**：单批次梯度噪声大，累积 50–200 步。
4. **用 HVP 替代 Adam 近似**：Adam 的 `exp_avg_sq` 是有偏的指数衰减估计，HVP 更忠实。
5. **在验证集上评估**：训练 loss 噪声大，验证 loss/困惑度信号更干净。
6. **尝试不同检查点**：训练中期的检查点（loss 曲面曲率更大）可能显示更大的二阶优势。
7. **确保 Hessian 计算正确**：HVP 用多个批次减少方差，平均 5–10 个批次。

## 让分布放松胜出的调参技巧

如果 Gamma 自适应分配没有明显优于均匀：

1. **测试多个全局稀疏率**：优势在中高稀疏率（20%–50%）最明显。
2. **验证 Gamma 拟合质量**：检查 KS 检验 p 值。如果某些层拟合差，尝试对数正态或 Weibull。
3. **检查逐层率方差**：如果最优率几乎均匀，说明模型各层同质——换一个架构多样性更大的模型（如 BERT-Large）。
4. **可视化得分分布**：画每层直方图确认异质性。形状差异大的层从自适应分配中获益最多。
5. **加入 magnitude-based 均匀剪枝**作为额外 baseline——这是最弱的 baseline，应该总是输。
6. **使用子层粒度**：不按层，而是按子层（Q, K, V, O, FFN1, FFN2, LayerNorm）分别拟合分布，捕获更多异质性。

## 可用模型和检查点

预微调检查点在 `/lihongliang/fangzl/ckpt-compress/checkpoints/`：
- `gpt2_small_wikitext103_1000steps/` — GPT-2 Small 在 WikiText-103 上微调
- `bert_large_sst2_1000steps/` — BERT-Large 在 SST-2 上微调
- `bert_large_mnli_1000steps/` — BERT-Large 在 MNLI 上微调
- `bert_large_stsb_1000steps/` — BERT-Large 在 STS-B 上微调

## 参考文件

- `references/codebase-guide.md` — 代码库导航和 API 参考
- `references/experiment-guide.md` — 实验设计和执行步骤
