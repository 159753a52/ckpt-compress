# 实验计划（修订版）

> 硬件约束：4× NVIDIA A30 (24GB)
> 策略调整：预训练成本过高，改为 fine-tuning + checkpoint compression

## 一、实验思路

用预训练好的模型做 fine-tuning，在训练过程中周期性地压缩 checkpoint 并从压缩 checkpoint 恢复，模拟容错训练场景。这与 Inshrinkerator 的 transfer learning 实验和 fault-tolerant training 实验设置一致。

核心流程：
```
正常fine-tune → checkpoint点 → 用不同方法压缩 → 从压缩checkpoint恢复 → 继续训练 → 评估最终质量
```

模拟多次故障恢复（5-10次），累积误差越大越能体现方法优势。

## 二、对比方法（Baselines）

| 方法 | 剪枝策略 | 率分配策略 | 说明 |
|------|---------|-----------|------|
| Magnitude | $\|w_i\|$ | Uniform | 最经典baseline |
| Inshrinkerator | $\|g_i \cdot w_i\|$（一阶） | 按层类型启发式 | 一阶importance + 启发式分配 |
| ExCP | 残差上的magnitude | 启发式公式 | 残差剪枝 |
| **Ours (Full)** | 完整damage score（一阶+二阶） | Distribution-aware | 完整方法 |

## 三、实验模型与数据集

### 优先级排序

| 优先级 | 模型 | 任务 | 数据集 | 评估指标 | 4×A30预估时间 |
|--------|------|------|--------|---------|--------------|
| **P0** | GPT-2 Medium (335M) | Language Modeling fine-tune | WikiText-103 | Perplexity | 3-6 小时 |
| **P1** | Pythia-410M (410M) | Instruction fine-tune | Alpaca (52K) | HellaSwag/ARC/PIQA Avg Acc | 2-4 小时 |
| **P2** | BERT-Large (345M) | NLU fine-tune | GLUE (SST-2/MNLI/STS-B) | Accuracy / Pearson Corr | 1-2 小时/任务 |
| **P3** | ViT-L/32 (307M) | Image Classification fine-tune | ImageNet-1K | Top-1 Accuracy | 6-12 小时 |

### 选择理由

- **GPT-2 Medium**：Figure 1 已使用该模型，保持全文一致性，最优先
- **Pythia-410M**：ExCP 的主要实验模型，可直接对比其公开结果
- **BERT-Large**：Inshrinkerator 做了 transfer learning 实验（GLUE），可直接对比
- **ViT-L/32**：覆盖 CV 领域，两篇对比论文都用了；用预训练权重 fine-tune 几十个 epoch 即可，不需要从头训练 300 epoch

最低要求完成 P0 + P1，覆盖两个 LLM 任务。P2 和 P3 按时间余量补充。

## 四、容错训练模拟设置

| 参数 | 设置 |
|------|------|
| 故障恢复次数 | 5-10 次（均匀分布在训练过程中） |
| Checkpoint 保存频率 | 每 N 步保存一次（N 根据总步数调整） |
| 压缩时机 | 每次保存 checkpoint 时压缩 |
| 恢复方式 | 从压缩 checkpoint 加载权重+优化器状态，继续训练 |
| 剪枝比例 | 50%, 70%, 80%, 90%（固定全局比例） |

## 五、图表产出清单

### 必须产出

| 编号 | 类型 | 内容 | 位置 |
|------|------|------|------|
| Fig 1 | 图 | Actual vs Predicted loss change（已有） | Introduction |
| Fig 2 | 图 | Overview 方法流程图（正在画） | Introduction / Method |
| Table 1 | 表 | **固定剪枝比例下的质量对比（主实验）** | Experiments |
| Table 2 | 表 | **联合压缩总压缩比对比**（剪枝+量化） | Experiments |
| Table 3 | 表 | **消融实验**（拆解两个贡献的独立增益） | Experiments / Analysis |
| Fig 3 | 图 | **Gamma拟合质量验证**（直方图+拟合PDF） | Experiments / Analysis |
| Fig 4 | 图 | **剪枝比例-质量 Pareto 曲线** | Experiments |

### 推荐产出

| 编号 | 类型 | 内容 | 位置 |
|------|------|------|------|
| Fig 5 | 图 | 层级剪枝率分布可视化（柱状图/热力图） | Analysis |
| Fig 6 | 图 | 训练恢复后的Loss曲线（多次恢复累积效果） | Experiments |
| Table 4 | 表 | 运行时开销对比 | Experiments |

### 可选（放附录）

- 不同分布族对比（Gamma vs Log-Normal vs Weibull）
- Bisection 收敛速度分析
- 更多模型/数据集的补充实验

## 六、各图表详细设计

### Table 1：固定剪枝比例下的质量对比（主实验，最核心）

每个模型一个子表或合并为一个大表。剪枝比例：50%、70%、80%、90%。

```
GPT-2 Medium fine-tune on WikiText-103 (Perplexity ↓)
| Method                  | 50%   | 70%   | 80%   | 90%   |
|-------------------------|-------|-------|-------|-------|
| Magnitude (Uniform)     |       |       |       |       |
| Inshrinkerator (1st)    |       |       |       |       |
| ExCP                    |       |       |       |       |
| Ours                    |       |       |       |       |

Pythia-410M fine-tune on Alpaca (下游任务 Avg Accuracy ↑)
| Method                  | 50%   | 70%   | 80%   | 90%   |
|-------------------------|-------|-------|-------|-------|
| ...                     |       |       |       |       |

BERT-Large fine-tune on GLUE (Accuracy ↑)
| Method                  | 50%   | 70%   | 80%   | 90%   |
|-------------------------|-------|-------|-------|-------|
| ...                     |       |       |       |       |
```

### Table 2：联合压缩对比

量化部分与其他论文一致（4-bit非均匀量化），只替换剪枝部分。

```
| Method              | CR    | Quality Deg. (%) |
|---------------------|-------|------------------|
| Inshrinkerator      |       |                  |
| ExCP                |       |                  |
| Ours + 4bit quant   |       |                  |
```

### Table 3：消融实验

拆解两个贡献：(1) 二阶信息的增益 (2) distributional allocation 的增益。

```
| Importance Score    | Rate Allocation  | 50%  | 70%  | 90%  |
|---------------------|------------------|------|------|------|
| Magnitude           | Uniform          |      |      |      |
| 1st-order           | Uniform          |      |      |      |
| 1st+2nd order       | Uniform          |      |      |      |
| 1st-order           | Dist-aware       |      |      |      |
| 1st+2nd order       | Dist-aware (Full)|      |      |      |
```

### Fig 3：Gamma拟合质量验证

- 选3个代表性层：Attention Q/K/V、MLP、LayerNorm
- 画 damage score 直方图（归一化为密度）
- 叠加拟合的 Gamma PDF 曲线
- 可加 KS 检验 p 值或拟合优度指标
- 子图排列：1行3列

### Fig 4：Pareto曲线

- X轴：剪枝比例（0% ~ 95%），Y轴：Perplexity 或 Accuracy
- 每个方法一条线：Magnitude / Inshrinkerator / ExCP / Ours
- 展示高剪枝比例下 Ours 优势更明显
- 每个模型一个子图，横向排列

### Fig 5：层级剪枝率分布

- 柱状图：X轴为层编号，Y轴为该层实际剪枝率
- 两组柱：Uniform（水平线）vs Ours（不同高度）
- 用颜色区分层类型（Attention / MLP / LayerNorm）

### Fig 6：训练恢复 Loss 曲线（关键实验）

- 模拟容错训练：多次从压缩 checkpoint 恢复
- X轴：训练步数，Y轴：Loss
- 多条线：No compression / Magnitude / Inshrinkerator / Ours
- 展示多次恢复后的累积误差差异
- 这个图对 fine-tuning 设置尤其重要，直接展示实际使用效果

## 七、实验执行优先级

1. **P0**：GPT-2 Medium fine-tune pipeline 跑通，产出 Table 1 一行 + Fig 6 的 loss 曲线
2. **P1**：Pythia-410M fine-tune on Alpaca，补全 Table 1
3. **P2**：消融实验（Table 3）+ Gamma 拟合验证（Fig 3）—— 可在 P0 的模型上直接做
4. **P3**：Pareto 曲线（Fig 4）+ 层级剪枝率分布（Fig 5）
5. **P4**：BERT-Large on GLUE（补充 NLU 任务）
6. **P5**：ViT-L/32 fine-tune on ImageNet（补充 CV 任务，如时间允许）
7. **P6**：联合压缩对比（Table 2）+ 运行时开销（Table 4）

## 八、论文表述建议

不需要强调"只做了fine-tuning"，可以这样写：

> "Following prior work on checkpoint compression (Inshrinkerator; ExCP), we evaluate our method in both fault-tolerant training and transfer learning scenarios. We simulate periodic checkpoint compression and restoration during fine-tuning of pretrained language models, measuring the accumulated quality degradation across multiple recovery cycles."

Inshrinkerator 本身就有 fault-tolerant training + transfer learning 两个实验场景，你的设置完全合理。

## 九、注意事项

- 剪枝比例的公平对比：所有方法在**相同全局剪枝比例**下对比
- ExCP 使用残差剪枝，对比时需说明设置差异（直接剪枝 vs 残差剪枝）
- 量化部分保持一致：所有方法使用相同的量化方案，只替换剪枝部分
- Diagonal Hessian 计算：使用 Fisher 信息矩阵的对角近似，fine-tuning 时梯度天然可用，开销很小
- 多次恢复实验是关键：累积误差是 checkpoint 压缩的核心挑战，多次恢复最能体现方法差异
- 4×A30 显存管理：GPT-2 Medium / Pythia-410M / BERT-Large 单卡都能放下，用数据并行加速即可
