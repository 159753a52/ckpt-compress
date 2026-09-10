# 单机双卡 (2x V100) 张量并行 (TP) 候选打分与检查点压缩实现方案

## 一、背景与目标

在单机双卡（2x V100）环境下，模型参数按张量并行（Tensor Parallelism, TP）拆分到两张卡上（GPU 0 与 GPU 1）。为解决大规模模型保存与打分时的显存膨胀与跨卡通信瓶颈，本项目实现轻量级**基于原生 PyTorch (`torch.distributed` + NCCL) 的 TP 候选打分机制**：
1. **显存轻量**：敏感度向量（VJP 梯度）计算后立即做本地点积并就地释放，显存开销从全量张量降至 $<2\%$，杜绝 V100 OOM。
2. **通信极简**：跨卡仅通过一次包含 $K$ 个浮点数的 `all_reduce`（如 3 套候选方案仅传 3 个 float = 12 字节），完成全局曲率收益汇聚。
3. **多方案共享**：单次反向探测同时评估 $\lambda \in \{0, 0.5, 1.0\}$（全局优先、层内平衡、混合平衡）3 套候选方案。

---

## 二、当前仓库与远程同步状态确认

- **代码仓库路径**：`code/checkpoint_compress`
- **远程分支**：`origin/codex/v100-experiments`（最新提交 `41b1103`）
- **本地分支**：已切换并关联至 `codex/v100-experiments`，与 `origin/codex/v100-experiments` **完全一致、工作区干净 (clean)**。

---

## 三、架构设计与模块划分

```
[Rank 0: GPU 0 (Slice W0, ΔW0)]       [Rank 1: GPU 1 (Slice W1, ΔW1)]
              │                                      │
              ├─── 同步生成/广播扰动探针 η (Pseudo-probe) ───┤
              │                                      │
       [Local VJP 反向]                        [Local VJP 反向]
       算得敏感度 g0                           算得敏感度 g1
              │                                      │
       [卡内流式候选点积]                       [卡内流式候选点积]
       β0[k] = ⟨g0, ΔW0 - Δθ0[k]⟩              β1[k] = ⟨g1, ΔW1 - Δθ1[k]⟩
       立即释放 g0 (del & empty_cache)         立即释放 g1 (del & empty_cache)
              │                                      │
              └─── NCCL All-Reduce (SUM) 仅传 K 个标量 ───┘
                                   │
                     全局候选收益 α[k] = β0[k] + β1[k]
                                   │
                      自适应选优 λ* = argmin α[k]
```

---

## 四、具体代码修改方案

### 1. [NEW] `experiments/lib/residual_tp.py`
实现单机多卡 TP 核心逻辑：
- `TPConfig`：封装 rank、world_size、device、process_group 配置。
- `shard_weights_tp()` / `unshard_weights_tp()`：按列并行（ColParallel，如 MLP fc1、QKV）与行并行（RowParallel，如 MLP proj、Attn out）对参数及残差 $\Delta W$ 进行切分。
- `tp_candidate_masks()`：结合分布式分位数/矩统计量，为每张卡生成本地候选掩码（支持 $\lambda \in [0, 1]$ 旋钮）。
- `compute_tp_candidate_projections()`：在各卡本地完成候选点积累加，并调用 NCCL 进行 $K$ 维标量向量的 `all_reduce(op=SUM)`。
- `streamed_tp_probe_scoring()`：逐层/逐块反向探测，反向完成后立即点积并释放大张量显存。

### 2. [MODIFY] `experiments/lib/distributed_stats.py`
扩展轻量级通信原语：
- 新增 `reduce_candidate_scalars(scores: torch.Tensor, process_group=None)`：执行 $K$ 标量 NCCL 归约，记录通信字节数（$4K$ 字节）与耗时。
- 增强现有 `reduce_score_moments` 对 TP 切片层形状的鲁棒性。

### 3. [NEW] `tests/test_residual_tp_scoring.py`
单元与集成测试：
- **切分重组一致性测试**：验证 ColParallel / RowParallel 切片与合并数学等价。
- **点积无损等价性测试**：验证两卡分布式标量归约结果与单卡全量点积严格一致（误差 $< 10^{-6}$）。
- **显存释放测试**：验证敏感度张量在点积后被即时回收，显存无泄露。
- **Mock/Gloo 测试**：在无多 GPU 环境下支持 Gloo/CPU 双进程模拟测试。

### 4. [NEW] `experiments/scripts/run_tp_v100_scoring.py`
实机两卡运行脚本：
- 支持启动命令：
  ```bash
  torchrun --nproc_per_node=2 experiments/scripts/run_tp_v100_scoring.py --model gpt2 --candidates 3 --device cuda
  ```
- 包含性能评测指标输出：单卡显存峰值、反向耗时、All-Reduce 通信耗时（微秒级）、候选打分准确率。
