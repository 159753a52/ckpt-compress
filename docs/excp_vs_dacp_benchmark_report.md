# 检查点压缩前沿基线对比：ExCP (ICML '24) vs DACP (二阶 HVP)

## 一、ExCP (ICML '24) 核心原理剖析

ExCP 是 2024 年 ICML 会议发表的大模型极限检查点压缩算法（*Extreme Checkpoint Compression for Large Language Models*），其核心思想是**复用优化器已有的二阶动量状态来辅助权重剪枝**：

1. **打分准则（Sensitivity Criterion）**：
   ExCP 认为单纯看权重变化量 $|\Delta W|$ 会误伤关键参数，因此引入 AdamW 的二阶矩 $m_t$（`exp_avg_sq`，即梯度的指数移动平均平方）：
   $$\text{Threshold } r_w = \frac{\alpha}{\sqrt{m_t} + \epsilon} \cdot \text{median}(|\Delta W|)$$
   等价于坐标重要性得分：
   $$\text{Score}_{\text{ExCP}} = |\Delta W| \cdot \sqrt{m_t}$$
   即：**权重变化量越大、且历史梯度方差越大（越活跃）的坐标，保留优先级越高**。
2. **计算特性**：
   直接利用训练时内存中现成的 Adam 状态，**无需进行额外的反向传播或 Hessian 探测**，计算几乎零开销。

---

## 二、实测环境与实验配置

- **加速卡**：NVIDIA Tesla V100-32GB GPU
- **模型**：GPT-2 预训练模型（1.24 亿参数，48 个可压缩权重矩阵）
- **训练任务**：WikiText-2 真实微调（50 个 AdamW optimizer steps，真实累积 $v_t$ 和 $m_t$ 优化器状态）
- **基准状态**：
  - 微调前基座 $W_0$：Val Loss = 3.6768, PPL = 39.52
  - 微调后目标 $W_1$（100% 全量）：Val Loss = 3.3172, PPL = 27.58
- **对比策略**：
  1. 传统幅值法（Magnitude）：$|\Delta W|$
  2. **ExCP 基准（ICML '24）**：$|\Delta W| \cdot \sqrt{m_t}$
  3. 一阶梯度法（First-Order）：$|g \cdot \Delta W|$
  4. **我们方法（DACP / 二阶 HVP）**：$|-g \cdot \Delta W + 0.5 \Delta W \cdot H \Delta W|$

---

## 三、实测结果横向对比（Tesla V100 实测数据）

### 1. 打分耗时开销对比

| 方法 | 额外反向传播批次 | 打分计算耗时 (ms) | 显存与计算特征 |
| :--- | :---: | :---: | :--- |
| **ExCP (ICML '24)** | **0 批次（零反向）** | **5.14 ms** | **极速**：直接点乘现有 Adam 优化器状态 |
| **DACP (二阶 HVP)** | 2 批次 | 438.7 ms | 包含 2 批次二阶曲率探测，开销依然在毫秒级 |

---

### 2. 各稀疏度下困惑度（PPL）对比表（越低越好）

*注：未压缩完整模型上限 PPL = **27.58**。*

| 压缩稀疏度 (Sparsity) | 传统幅值法 PPL | **ExCP (ICML '24) PPL** | 一阶梯度 PPL | **DACP (二阶 HVP) PPL** | ExCP 相对幅值法优势 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **50% 压缩** | 27.58 | **27.61** | 27.88 | **27.80** | 近乎完全无损 |
| **70% 压缩** | 27.78 | **27.80** | 28.46 | **28.28** | 保持高保真 |
| **80% 压缩** | 28.19 | **28.14** | 29.14 | **28.88** | PPL 优于幅值 0.05 |
| **90% 压缩** | 29.39 | **29.11** | 30.54 | **30.20** | **PPL 优于幅值 0.28** |
| **95% 压缩** | 30.93 | **30.36** | 31.94 | **31.60** | **PPL 优于幅值 0.57** |

---

## 四、核心深度洞察

### 1. ExCP 的优势机理
- **时序累积优势**：ExCP 借助 Adam 的 $m_t$，吸收了微调全流程（50 步）每一布的梯度信息。
- **抗噪性好**：在高稀疏度（90%～95%）下，单纯看 $|\Delta W|$ 容易保留偶然跳变的大坐标，而 ExCP 乘以历史方差 $\sqrt{m_t}$，优先剔除了偶发漂移权重，因此 **ExCP 表现严格优于纯幅值法**。

### 2. DACP 与 ExCP 的互补与本质区别
- **依赖性差异**：
  - ExCP **强绑定 Adam/AdamW 优化器**。一旦训练使用 SGD、Adafactor、分布式分片优化器（ZeRO/FSDP）或者用户仅提供模型权重 checkpoint（无优化器状态）时，ExCP 无法计算；
  - DACP **独立于优化器**，对任意检查点仅需 2 批次校准前向反向即可完成全局二阶打分。
- **非对角曲率（跨参数耦合）**：
  - ExCP 仅考虑了单个参数维度的历史二阶矩（对角 Fisher 近似）；
  - DACP 的 HVP 包含了完整的 Hessian 矩阵与候选扰动向量的全局点积，能感知跨层、跨注意力的参数协同作用。

---

## 五、复现命令与产物

- **对比脚本**：[`experiments/scripts/eval_compare_excp.py`](file:///D:/Paper/EMNLP_26_reorganized/code/checkpoint_compress/experiments/scripts/eval_compare_excp.py)
- **原始测试数据**：[`experiments/results/v100_excp_comparison.json`](file:///D:/Paper/EMNLP_26_reorganized/code/checkpoint_compress/experiments/results/v100_excp_comparison.json)
- **运行命令**：
  ```bash
  source /root/venv/bin/activate
  cd /root/ckpt-compress
  python3 experiments/scripts/eval_compare_excp.py --device cuda
  ```
