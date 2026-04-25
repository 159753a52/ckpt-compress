# DACP 复杂度分析与实验时间预估

## 模型参数量

| 模型 | 总参数量 N | 可剪枝参数 P | 层数 L | 显存占用 |
|------|-----------|------------|--------|---------|
| GPT-2 Small | 124M | ~100M | 12 | ~2 GB |
| GPT-2 Medium | 355M | ~300M | 24 | ~4 GB |
| Pythia-410M | 410M | ~340M | 24 | ~4.5 GB |
| BERT-Large | 340M | ~280M | 24 | ~4 GB |
| ViT-L/32 | 305M | ~250M | 24 | ~3.5 GB |
| ResNet-18 | 11M | ~10M | 18 | ~1 GB |

## 核心操作复杂度

### 1. 重要性评分计算

> 注意：`second-order-hvp` 实际上是**完整的二阶泰勒展开**，包含一阶项 (-g·θ) 和二阶项 (α·θ·Hθ)，不是纯二阶方法。

| 方法 | 公式 | 时间复杂度 | 空间复杂度 | 说明 |
|------|------|-----------|-----------|------|
| **Magnitude** | \|w\| | O(P) | O(P) | 零阶，无前向/反向传播 |
| **First-order** | \|g·w\| | O(B·P + B·F) | O(P) | 一阶泰勒，B 个 batch 的梯度累积 |
| **Second-order HVP (full)** | \|-g·θ + α·θ·Hθ\| | O(B·(2F + 2P)) | O(2P) | 一阶+二阶泰勒，使用真实 HVP |
| **Second-order HVP (block)** | 同上 | O(⌈L/C⌉·B·(2F + 2P')) | O(2P' + P) | 分块计算，节省显存 |
| **Residual magnitude** | \|W_t - W_ref\| | O(P) | O(2P) | 需加载参考权重 |

其中：
- B = `--hvp_batches`（默认 8）
- F = 单次前向传播代价 ≈ O(N·S)，S 为序列长度
- C = `--chunk_size`（默认 10）
- P' = P/⌈L/C⌉ 每块参数量

**关键瓶颈**：HVP 的二次反向传播需要保留完整计算图（`create_graph=True`），显存占用约为普通前向的 2-3 倍。

### 2. 分配策略

| 策略 | 时间复杂度 | 空间复杂度 | 说明 |
|------|-----------|-----------|------|
| **Uniform** | O(L) | O(L) | 直接赋值 |
| **Gamma-adaptive** | O(L·S_l + I·L) | O(L) | 每层矩估计 O(S_l)，二分法 I 次迭代（I≤200） |
| **Global TopK** | O(P log P) 或 O(P·K_s) | O(P) 或 O(K_s) | 全排序或采样 K_s=100K 后排序 |
| **Per-type** | O(P) | O(P) | 按 type 分组后阈值分配 |

其中 S_l 为第 l 层参数数，在大模型（P > 10M）时 GlobalTopK 使用采样估计。

### 3. 剪枝执行

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| `apply_pruning` | O(P) | O(P)（存储 mask） |
| `kthvalue` 每层 | O(S_l) 期望 | O(S_l) |

### 4. 量化

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| INT4 对称量化 | O(P) | O(P)（int8 存储） |
| INT4 反量化 | O(P) | O(P) |
| K-Means 量化 (ExCP) | O(P·K·I_km) | O(P) |
| Approx K-Means (Inshr.) | O(P + H·K·I_km) | O(H + K) |

K = 量化级别（16），I_km = K-Means 迭代次数（~100），H = 直方图桶数。

### 5. 评估

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| LM 评估 (perplexity) | O(E·N·S) | O(N) |
| CLS/CV 评估 (accuracy) | O(E·N·S) | O(N) |
| lm_eval 下游评估 | O(T·E_t·N·S) | O(N) |

E = `--eval_batches`，T = 下游任务数，E_t = 每任务样本数。

## 实验脚本复杂度分析

### Table 1: `run_method_comparison.py`

**最耗时步骤**：预计算所有方法的 importance scores

```
Score 计算:
  magnitude:        O(P)                    ~1 秒
  first-order:      O(B·(F+P))              ~30 秒 (B=8)
  residual-mag:     O(P)                    ~1 秒
  second-order-hvp: O(α_n · B·(2F+2P))     ~5-10 分钟 (每个 alpha 值)

剪枝+评估 (per method × ratio):
  每个配置: O(P + E·N·S)                   ~15 秒

总组合数: 6 methods × 4 ratios × α_n alpha_values = 24-48 次评估
```

| 模型 | Score 计算 | 剪枝评估 | 总计 |
|------|-----------|---------|------|
| GPT-2 Small | ~5 min | ~6 min | **~11 min** |
| GPT-2 Medium | ~20 min | ~15 min | **~35 min** |
| Pythia-410M | ~25 min | ~18 min | **~43 min** |

### Table 2: `run_joint_compression.py`

与 Table 1 类似但每个配置额外做一次量化+评估。

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~25 min** |

### Table 3: `run_ablation_study.py`

6 种消融 × 4 剪枝率，共 24 次评估。Score 计算与 Table 1 共享。

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~30 min** |

### Figure 4: `run_pareto_curves.py`

4 methods × 9 ratios (0.1-0.9) = 36 次评估。

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~25 min** |

### Figure 5: `run_pruning_heatmap.py`

默认 1 method × 1 ratio，只需 1 次 score + 1 次分配。

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~8 min** |

### Figure 6: `run_fault_tolerant_training.py`

4 methods × (训练 + 反复压缩)。这是 **最耗时** 的实验。

每个方法需要完整的多段训练 + 多次在线压缩。

```
单方法:
  训练: num_steps_per_seg × (num_recoveries + 1) 步
  压缩: num_recoveries 次 (每次: 梯度收集 + Pruner score + allocate + prune)
```

| 模型 | 单方法 | 4 方法总计 |
|------|--------|-----------|
| GPT-2 Medium | ~30 min | **~2 hours** |

### §9.1-9.2: `run_gamma_validation.py`

1 次 score 计算 + KS 检验（纯 CPU 数值计算, 快速）+ 3 策略 × 3 剪枝率 = 9 次评估。

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~15 min** |

### §9.3: `run_sensitivity_analysis.py`

- alpha 扫描：5-10 值 × 3 剪枝率 = 15-30 次评估
- K 值扫描（sweep_K）：5 K 值 × 2 方法 × 完整训练 = **耗时**
- timing breakdown：4 阶段计时

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~1.5 hours** |

### §9.4: `run_layer_distribution_violin.py`

1 次 score 计算 + DataFrame 构建 + 绘图。

| 模型 | 总计 |
|------|------|
| GPT-2 Medium | **~10 min** |

## 全套实验时间汇总

### 单模型 (GPT-2 Medium, wikitext103, 单 GPU)

| 实验 | 预估时间 |
|------|---------|
| Table 1 | ~35 min |
| Table 2 | ~25 min |
| Table 3 | ~30 min |
| Fig 4 | ~25 min |
| Fig 5 | ~8 min |
| Fig 6 | ~2 hours |
| §9.1-9.2 Gamma | ~15 min |
| §9.3 Sensitivity | ~1.5 hours |
| §9.4 Violin | ~10 min |
| **单模型合计** | **~5.5 hours** |

### 论文完整实验（3 模型）

| 模型 | 单模型时间 |
|------|-----------|
| GPT-2 Medium | ~5.5 hours |
| Pythia-410M | ~6.5 hours |
| ViT-L/32 + CIFAR-10 | ~3 hours |
| **三模型合计** | **~15 hours** |

## 显存使用分析

| 阶段 | GPT-2 Small | GPT-2 Medium | Pythia-410M |
|------|------------|-------------|------------|
| 模型加载 | ~0.5 GB | ~1.4 GB | ~1.6 GB |
| 前向传播 | ~1.5 GB | ~3 GB | ~3.5 GB |
| 梯度计算 | ~2 GB | ~4.5 GB | ~5 GB |
| HVP (full mode) | ~4 GB | ~8 GB | ~9 GB |
| HVP (block, chunk=10) | ~3 GB | ~6 GB | ~7 GB |
| HVP (block, chunk=5) | ~2.5 GB | ~5 GB | ~6 GB |

**推荐配置**：
- 8 GB VRAM：GPT-2 Small (full) 或 GPT-2 Medium (block, chunk=5)
- 16 GB VRAM：GPT-2 Medium (full) 或 Pythia-410M (block, chunk=10)
- 24 GB VRAM：所有模型 full mode

## 加速建议

1. **快速验证**：先用 `gpt2-small --num_steps 10 --eval_batches 5` 跑 Table 1 确认流程
2. **跳过耗时实验**：`--skip_timing` 可跳过 §9.3 的 timing breakdown
3. **减少 HVP batch**：`--hvp_batches 4` 减半 HVP 计算时间（精度略降）
4. **block mode**：`--hvp_mode block --chunk_size 5` 大幅降低显存需求
5. **并行跑不同模型**：Table 1 不同模型间无依赖，可在多卡上并行
