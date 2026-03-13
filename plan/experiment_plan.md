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
| Magnitude | |w_i| | Uniform（固定） | 最弱 baseline |
| Inshrinkerator | |g_i · w_i|（一阶） | Uniform（固定） | 一阶 baseline（原论文方法） |
| ExCP | |w_t - w_{t-1}|（残差 magnitude） | Uniform（固定） | 残差 magnitude baseline |
| Ours (score only) | damage score（一阶+二阶） | Uniform（固定） | 仅替换 importance |
| **Ours (Full)** | **damage score（一阶+二阶）** | **Gamma-adaptive (per-layer)** | **完整方法** |

### Inshrinkerator 剪枝策略说明

**Inshrinkerator 的剪枝比例分配本质上是 Uniform Pruning**：
- 搜索空间中 Pruning Fraction 只有一个全局参数 F，候选值 = {0, 0.1, 0.2, 0.3, 0.4, 0.5}
- 除 Embedding 层外，所有层都独立剪掉自己最不重要的 F% 参数（每层内部按 importance 排序）
- 不同层类型（Conv / Linear / Attention）之间没有差异化的比例分配
- F 作为配置立方体的一个轴，与量化 bins、保护比例一起联合搜索，目标是 max CR s.t. quality loss ≤ ε
- 论文中 "per-layer type pruning" 的含义是每层独立排序剪 F%（而非跨层汇集排序），叫法是为了与 global pruning 区分

**因此 Table 1 的分配基线使用 Uniform**：给定全局剪枝比例，所有层（除 Embedding）用相同比例，各层内部按 importance 排序剪枝。这与 Inshrinkerator 原论文方法一致。后续实验只需搜索一个合理的全局 F 值即可。

**关键论点**：Ours (score only) vs Inshrinkerator = 相同 Uniform 分配，不同得分 → 证明得分更优。
Ours (Full) vs Ours (score only) = 相同得分，不同分配 → 证明 Gamma-adaptive 更优（从 Uniform 到 per-layer adaptive）。

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
| 剪枝比例 | 10%, 20%, 30%, 40%（固定全局比例） |

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
| Fig 5 | 图 | **剪枝率分配对比热力图**（Ours vs Inshrinkerator search，per-type + per-layer 双粒度） | Analysis |
| Fig 6 | 图 | 训练恢复后的Loss曲线（多次恢复累积效果） | Experiments |
| Fig 7 | 图 | **分配开销 Scaling 图**（X=模型参数量，Y=分配优化时间，Ours 几乎为 0） | Experiments |
| Table 4 | 表 | **运行时间分解对比**（Breakdown：得分计算 + 分配优化 + 总计） | Experiments |

### 可选（放附录）

- 不同分布族对比（Gamma vs Log-Normal vs Weibull）
- Bisection 收敛速度分析
- 更多模型/数据集的补充实验

## 六、各图表详细设计

### Table 1：固定剪枝比例下的质量对比（主实验，最核心）

每个模型一个子表或合并为一个大表。剪枝比例：10%、20%、30%、40%。

**Table 结构（以 GPT-2 Medium 为例）**：

```
GPT-2 Medium fine-tune on WikiText-103 (Perplexity ↓)
| Method                              | 10%   | 20%   | 30%   | 40%   |
|--------------------------------------|-------|-------|-------|-------|
| Magnitude + per-type                 |       |       |       |       |
| Inshrinkerator (1st-order + per-type)|       |       |       |       |
| ExCP (residual-mag + per-type)       |       |       |       |       |
| Ours (damage score + per-type)       |       |       |       |       |
| Ours (damage score + Gamma-adaptive) |       |       |       |       |
```

**注意**：前 4 行共享相同的 Uniform 分配（与 Inshrinkerator 原论文一致，所有层用同一个 F%），仅改变 importance score。
Row 4 vs Row 5：相同 importance，不同分配策略（Uniform vs Gamma-adaptive）。

**Uniform 比例设置**：直接指定全局剪枝比例 F（如 30%/50%/70%/90%），所有层（除 Embedding）各自独立按 importance 排序剪掉 F% 的参数。

### Table 2：联合压缩对比（剪枝 + 量化）

**我们的方法核心在剪枝**，量化部分复用 Inshrinkerator 的方案，保持一致性。

**量化方案（可开关复用）**：
- 使用 Inshrinkerator 的 DDSketch 加速 K-Means 非均匀量化
- DDSketch 将参数投影到对数空间构建直方图，然后在直方图 bins 上做加权 K-Means（而非原始参数），加速 ~65x
- 量化 bins 数量从搜索空间 {4, 6, 8, 12, 16, 32} 中选取
- Embedding 层单独设置更保守的 bins（{16, 32}）
- 设计为**开关模式**：`--enable_quantization` 开启时自动应用量化，关闭时只做剪枝
- 量化模块独立于剪枝模块，可与任意剪枝方法组合复用

```
| Method              | Pruning | Quantization        | CR    | Quality Deg. (%) |
|---------------------|---------|---------------------|-------|------------------|
| Inshrinkerator      | 1st-order + Uniform | DDSketch K-Means | | |
| ExCP                | residual-mag | 同上               |       |                  |
| Ours                | damage score + Gamma | 同上（复用）     |       |                  |
```

**论文叙事**：量化方案保持一致（均使用 Inshrinkerator 的 DDSketch K-Means），差异仅在剪枝部分，确保对比公平。

### Table 3：消融实验

拆解两个贡献：(1) 二阶信息的增益 (2) distributional allocation 的增益。

```
| Importance Score    | Rate Allocation     | 10%  | 20%  | 30%  | 40%  |
|---------------------|---------------------|------|------|------|------|
| Magnitude           | per-type (固定)     |      |      |      |      |
| 1st-order           | per-type (固定)     |      |      |      |      |
| 1st+2nd order       | per-type (固定)     |      |      |      |      |
| Magnitude           | Gamma-adaptive      |      |      |      |      |
| 1st-order           | Gamma-adaptive      |      |      |      |      |
| 1st+2nd order       | Gamma-adaptive      |      |      |      |      |
```

这是一个 3×2 的消融矩阵，能清晰分离各因素贡献。

### Fig 3：Gamma拟合质量验证

- 选3个代表性层：Attention Q/K/V、MLP、LayerNorm
- 画 damage score 直方图（归一化为密度）
- 叠加拟合的 Gamma PDF 曲线
- 可加 KS 检验 p 值或拟合优度指标
- 子图排列：1行3列

### Fig 4：Pareto曲线

- X轴：剪枝比例（0% ~ 40%），Y轴：Perplexity 或 Accuracy
- 每个方法一条线：Magnitude / Inshrinkerator / ExCP / Ours
- 展示 Ours 在各剪枝比例下的优势
- 每个模型一个子图，横向排列

### Fig 5：剪枝率分配对比热力图（核心 efficiency 论证）

**核心论点**：Inshrinkerator 使用 Uniform 分配（所有层同一个 F%），而我们的 Gamma-adaptive 方法能自动为每层分配不同的剪枝比例，且计算开销几乎为零。

**热力图设计**：
- 热力图：行=层编号，列=全局稀疏率（10%/20%/30%/40%），颜色=该层剪枝率
- 用颜色条区分层类型（Attention / MLP / LayerNorm）
- 展示 Gamma-adaptive 为不同层分配了差异化的剪枝比例
- 叠加 Inshrinkerator 的 Uniform 分配作为水平参考线（所有层一条直线）
- 对比效果：Uniform 是一条平线，Gamma-adaptive 是有高低起伏的分布

**论文叙事**：Inshrinkerator 给所有层同样的剪枝比例（Uniform），无法捕获层间异质性。Gamma-adaptive 通过分布拟合自动识别每层的冗余程度，为冗余层分配更高剪枝率、敏感层更低剪枝率，且不需要任何搜索开销（O(1) vs Inshrinkerator 的多次前向评估搜索）。

### Fig 6：训练恢复 Loss 曲线（关键实验）

- 模拟容错训练：多次从压缩 checkpoint 恢复
- X轴：训练步数，Y轴：Loss
- 多条线：No compression / Magnitude / Inshrinkerator / Ours
- 展示多次恢复后的累积误差差异
- 这个图对 fine-tuning 设置尤其重要，直接展示实际使用效果

### Table 4：运行时间分解对比（Breakdown Table）

**核心论点**：分配优化是 Inshrinkerator 的瓶颈，我们几乎为零。

```
GPT-2 Medium (335M params, 24 layers)
| 阶段                    | Inshrinkerator       | Ours                |
|------------------------|----------------------|---------------------|
| 得分计算               | gradient EMA (50 batches) | gradient + HVP (N batches) |
| 得分计算时间            | ~X s                 | ~Y s                |
| 分配优化               | Grid search (M次前向) | Gamma fit + bisection |
| 分配优化时间            | ~X s (M次前向评估)   | ~0.01 s             |
| 分配评估次数            | M 次模型前向          | 0 次                |
| 总计                    | ~X s                 | ~Y s                |

BERT-Large (345M params, 24 layers)
| ...                    | ...                  | ...                 |
```

**关键数字**：
- Inshrinkerator 的配置立方体搜索：bins(6) × pruning(6) × protection(3) × metric(2) = 216 种组合
  - 引导式穷举搜索利用单调性大幅减少评估次数，但仍需 ~20-50 次前向评估
  - 后续 checkpoint 用邻域搜索，~5-10 次前向评估
  - 注意：Inshrinkerator 的剪枝比例是 Uniform 的（所有层同一个 F%），搜索的是最优 F 值
- 我们的 Gamma fit + bisection：纯数值计算，~10ms，0 次前向评估
- 差距：**数千倍**

**每次 checkpoint 都要做**：在 fault-tolerant 场景中，10 次恢复 = 10 次搜索，累计开销差距更大。

### Fig 7：分配优化 Scaling 图

- X轴：模型参数量（对数刻度，100M → 1B → 10B）
- Y轴：分配优化时间（对数刻度）
- 两条线：Inshrinkerator search（随参数量/层数增长，搜索空间指数扩大）vs Ours（几乎水平）
- 底部标注"N× forward evals" vs "1× bisection"
- 如果实际跑不了大模型，可以用理论 FLOPs 外推

**论文叙事**：随着模型规模增大，Inshrinkerator 的搜索成本线性增长（每次前向 O(N)，搜索 M 次 = O(MN)），而我们的 Gamma fit + bisection 是 O(N)（只遍历一次得分计算均值方差）+ O(L·log(1/eps))（bisection，L 是层数），与模型参数量几乎无关。

## 七、实验执行优先级

1. **P0**：GPT-2 Medium 上跑通 Table 1 全流程
   - 设定 Uniform 剪枝比例（10%/20%/30%/40%），与 Inshrinkerator 一致
   - 计算所有 importance scores（magnitude / first-order / second-order-hvp / residual-magnitude）
   - 跑 Table 1 的 6 行 × 4 个剪枝率 = 24 个实验点
2. **P1**：GPT-2 Medium 上完成 Fig 5 热力图 + Table 4/Fig 7 运行时间对比
   - 热力图数据来自 P0 的 Gamma-adaptive 结果 vs Uniform 水平线
   - 运行时间 benchmark 可同步进行
3. **P2**：消融实验（Table 3）+ Gamma 拟合验证（Fig 3）—— 在 P0 模型上直接做
4. **P3**：Pareto 曲线（Fig 4）—— P0 数据的多 ratio 可视化
5. **P4**：Fig 6 容错训练 loss 曲线 —— 需要实际的多次恢复训练循环
6. **P5**：BERT-Large on GLUE / Pythia-410M on Alpaca（补全 Table 1 其他模型）
7. **P6**：ViT-L/32 fine-tune on ImageNet（CV 任务，时间允许时）
8. **P7**：联合压缩对比（Table 2）

## 八、论文表述建议

不需要强调"只做了fine-tuning"，可以这样写：

> "Following prior work on checkpoint compression (Inshrinkerator; ExCP), we evaluate our method in both fault-tolerant training and transfer learning scenarios. We simulate periodic checkpoint compression and restoration during fine-tuning of pretrained language models, measuring the accumulated quality degradation across multiple recovery cycles."

Inshrinkerator 本身就有 fault-tolerant training + transfer learning 两个实验场景，你的设置完全合理。

## 九、注意事项

- **公平分配**：Table 1 的前 4 行共享 Uniform 剪枝分配（与 Inshrinkerator 一致），仅改变 importance score
- **Uniform 分配与 Inshrinkerator 一致**：Inshrinkerator 原论文就是 Uniform pruning（所有层用同一个 F%），我们在 Table 1 的 baseline 对比中也用 Uniform，确保公平
- ExCP 使用残差剪枝（W_t - W_{t-1}），Table 1 中需说明设置差异。残差剪枝依赖前一检查点，增加存储复杂性
- 量化部分保持一致：所有方法复用 Inshrinkerator 的 DDSketch K-Means 量化方案，只替换剪枝部分。量化为可开关模块（`--enable_quantization`），Table 1 只看剪枝，Table 2 开启量化看联合压缩比
- Diagonal Hessian 计算：通过 HVP 精确计算，不再使用 Adam exp_avg_sq 近似
- **Inshrinkerator 搜索开销**：Inshrinkerator 通过配置立方体搜索（bins × F × protection）联合优化，需要多次前向评估。虽然剪枝比例本身是 Uniform 的，但搜索最优 F 值仍需开销。Table 4 和 Fig 7 要明确呈现这个差距
- 多次恢复实验是关键：累积误差是 checkpoint 压缩的核心挑战，多次恢复最能体现方法差异
- 4×A30 显存管理：GPT-2 Medium / Pythia-410M / BERT-Large 单卡都能放下，用数据并行加速即可
