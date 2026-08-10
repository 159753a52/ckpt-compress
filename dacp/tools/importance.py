"""
DACP 重要性得分计算模块。

实现参数重要性得分计算：

HVP（Hessian-Vector Product）精确计算：
   s_i = -g_i * θ_i + α * θ_i * (H * θ)_i
   - 使用真实的 Hessian 矩阵
   - 考虑参数之间的相互作用（非对角元素）

理论依据：删除参数 θ_i（设为0）对损失的影响，泰勒展开为：
ΔL ≈ g^T·Δθ + 0.5·Δθ^T·H·Δθ

当 Δθ = -θ（参数被剪枝到0）时：
ΔL ≈ -g^T·θ + 0.5·θ^T·H·θ

分解到每个参数的贡献：
s_i = -g_i·θ_i + 0.5·θ_i·(H·θ)_i

一阶项保留符号：
- 若 -g_i·θ_i > 0：删除该参数会增加损失（参数重要）
- 若 -g_i·θ_i < 0：删除该参数会减少损失（参数可能有害）
"""

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from typing import Dict, List, Callable


def compute_importance_scores_magnitude(
    weights: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    计算每个参数的重要性得分（基于权重绝对值）。

    公式: d_i = |θ_i|

    参数:
        weights: 模型权重字典 {name: tensor}

    返回:
        重要性得分字典 {name: tensor}
    """
    scores = {}

    for name, weight in weights.items():
        scores[name] = torch.abs(weight)

    return scores


def get_flattened_scores(
    scores: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """
    将所有层的重要性得分展平为一维张量。

    参数:
        scores: 重要性得分字典 {name: tensor}

    返回:
        展平后的一维张量
    """
    if not scores:
        return torch.tensor([])

    flattened = []
    for tensor in scores.values():
        flattened.append(tensor.flatten())

    return torch.cat(flattened)


def compute_mean_importance(
    scores: Dict[str, torch.Tensor]
) -> float:
    """
    计算平均重要性得分 s̄ = (1/N) * Σs_i。

    参数:
        scores: 重要性得分字典 {name: tensor}

    返回:
        平均重要性得分
    """
    flat = get_flattened_scores(scores)
    if flat.numel() == 0:
        return 0.0
    return flat.mean().item()


# ============================================================
# HVP (Hessian-Vector Product) 相关函数
# ============================================================

def compute_hvp(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batch: Dict[str, torch.Tensor],
    vector: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    计算 Hessian-Vector Product: H * v

    使用两次反向传播计算 HVP，不需要显式构建 Hessian 矩阵。
    原理：
        H * v = ∂/∂θ (∇L · v)

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数，接收 model 和 data_batch，返回标量损失
        data_batch: 数据批次字典（如 {"input_ids": tensor, "attention_mask": tensor}）
        vector: 向量字典 {name: tensor}，与模型参数同形状

    返回:
        HVP 结果字典 {name: tensor}
    """
    # 确保模型参数需要梯度
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}

    # 使用 MATH backend 回退，因为 Flash/Efficient Attention 不支持二阶导
    with sdpa_kernel(SDPBackend.MATH):
        # 第一次前向传播和反向传播，计算梯度
        model.zero_grad()
        loss = loss_fn(model, data_batch)

        # 计算一阶梯度，保留计算图
        grads = torch.autograd.grad(
            loss,
            list(params.values()),
            create_graph=True,
            retain_graph=True,
        )

        # 计算 grad · vector 的标量积
        grad_vector_product = torch.tensor(0.0, device=loss.device)
        for g, name in zip(grads, params.keys()):
            if name in vector:
                grad_vector_product = grad_vector_product + (g * vector[name]).sum()

        # 第二次反向传播，计算 HVP
        hvp_result = torch.autograd.grad(
            grad_vector_product,
            list(params.values()),
            retain_graph=False,
        )

    # 转换为字典格式
    hvp_dict = {
        name: hvp.detach()
        for name, hvp in zip(params.keys(), hvp_result)
    }

    return hvp_dict


def compute_hvp_batched(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    vector: Dict[str, torch.Tensor],
    num_batches: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    使用多个批次计算平均 HVP，提高估计稳定性。

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数
        data_batches: 数据批次列表
        vector: 向量字典
        num_batches: 使用的批次数量

    返回:
        平均 HVP 结果字典
    """
    hvp_sum = None
    actual_batches = min(num_batches, len(data_batches))

    for i in range(actual_batches):
        batch = data_batches[i]
        hvp = compute_hvp(model, loss_fn, batch, vector)

        if hvp_sum is None:
            hvp_sum = {name: h.clone() for name, h in hvp.items()}
        else:
            for name in hvp_sum:
                hvp_sum[name] += hvp[name]

    # 计算平均
    if hvp_sum is None:
        return {}

    hvp_avg = {
        name: h / actual_batches
        for name, h in hvp_sum.items()
    }

    return hvp_avg


def compute_importance_scores_hvp(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    num_batches: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    使用 HVP 计算参数重要性得分。

    公式: s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i

    其中：
    - g_i: 参数的梯度
    - θ_i: 参数值
    - (H * θ)_i: Hessian-Vector Product 的第 i 个元素

    相比 Adam 二阶矩近似，HVP 方法：
    - 考虑了 Hessian 的非对角元素（参数间相互作用）
    - 计算更准确，但速度较慢
    - 需要两次反向传播

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数，签名为 loss_fn(model, batch) -> scalar
        data_batches: 数据批次列表
        num_batches: 用于计算 HVP 的批次数量（更多批次 = 更稳定的估计）

    返回:
        重要性得分字典 {name: tensor}

    示例:
        >>> def loss_fn(model, batch):
        ...     outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
        ...     return outputs.loss
        >>> scores = compute_importance_scores_hvp(model, loss_fn, batches, num_batches=5)
    """
    # 收集模型参数（作为向量 v = θ）
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone() for name, p in params.items()}

    # 计算梯度（用于一阶项）
    model.zero_grad()
    loss = loss_fn(model, data_batches[0])
    loss.backward()

    gradients = {
        name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in params.items()
    }

    # 计算 HVP: H * θ
    print("[HVP] 计算 Hessian-Vector Product...")
    hvp_result = compute_hvp_batched(
        model, loss_fn, data_batches, weights, num_batches
    )

    # 计算重要性得分: s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i
    scores = {}
    for name in weights:
        theta = weights[name]
        grad = gradients.get(name, torch.zeros_like(theta))
        hvp = hvp_result.get(name, torch.zeros_like(theta))

        # 一阶项: -g * θ
        first_order = -grad * theta

        # 二阶项: 0.5 * θ * (H * θ)
        second_order = 0.5 * theta * hvp

        scores[name] = first_order + second_order

    return scores


def compute_importance_scores_hvp_abs(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    num_batches: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    使用 HVP 计算参数重要性得分（绝对值版本）。

    公式: d_i = |g_i * θ_i - 0.5 * θ_i * (H * θ)_i|

    这个版本使用绝对值，避免正负值相互抵消，
    使得分布更宽，更容易解释。

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数，签名为 loss_fn(model, batch) -> scalar
        data_batches: 数据批次列表
        num_batches: 用于计算 HVP 的批次数量（更多批次 = 更稳定的估计）

    返回:
        重要性得分字典 {name: tensor}

    示例:
        >>> def loss_fn(model, batch):
        ...     outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
        ...     return outputs.loss
        >>> scores = compute_importance_scores_hvp_abs(model, loss_fn, batches, num_batches=5)
    """
    # 收集模型参数（作为向量 v = θ）
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone() for name, p in params.items()}

    # 计算梯度（用于一阶项）
    model.zero_grad()
    loss = loss_fn(model, data_batches[0])
    loss.backward()

    gradients = {
        name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in params.items()
    }

    # 计算 HVP: H * θ
    print("[HVP] 计算 Hessian-Vector Product...")
    hvp_result = compute_hvp_batched(
        model, loss_fn, data_batches, weights, num_batches
    )

    # 计算重要性得分: d_i = |g_i * θ_i - 0.5 * θ_i * (H * θ)_i|
    scores = {}
    for name in weights:
        theta = weights[name]
        grad = gradients.get(name, torch.zeros_like(theta))
        hvp = hvp_result.get(name, torch.zeros_like(theta))

        # 一阶项: -g * θ
        first_order = -grad * theta

        # 二阶项: 0.5 * θ * (H * θ)
        second_order = 0.5 * theta * hvp

        # 使用绝对值
        scores[name] = torch.abs(first_order + second_order)

    return scores


def build_transformer_blocks(
    model: torch.nn.Module,
    model_family: str = 'gpt2',
) -> List[List[str]]:
    """
    将模型参数按 Transformer layer 分组为 block。

    每个 block 对应一个 Transformer layer，包含该层所有可训练的权重矩阵
    （Attention Q/K/V/O + MLP up/down 等）。1D 参数（bias、LayerNorm）
    不参与 block-wise HVP，但仍会被包含在对应 block 中以确保完整覆盖。

    参数:
        model: PyTorch 模型
        model_family: 模型族名称，用于确定参数名中的层号模式
            支持: 'gpt2', 'pythia', 'vit', 'bert'

    返回:
        List[List[str]]，每个元素是一个 block 内的参数名列表，按层号排序
    """
    import re

    layer_patterns = {
        'gpt2':   r'(?:transformer\.)?h\.(\d+)\.',
        'pythia': r'gpt_neox\.layers\.(\d+)\.',
        'vit':    r'(?:vit\.)?encoder\.layer\.(\d+)\.',
        'bert':   r'(?:bert\.)?encoder\.layer\.(\d+)\.',
    }

    if model_family not in layer_patterns:
        raise ValueError(
            f"Unknown model_family: {model_family}. "
            f"Available: {list(layer_patterns.keys())}"
        )

    pattern = re.compile(layer_patterns[model_family])
    blocks: Dict[int, List[str]] = {}

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        match = pattern.search(name)
        if match:
            layer_idx = int(match.group(1))
            blocks.setdefault(layer_idx, []).append(name)

    if not blocks:
        raise RuntimeError(
            f"No Transformer layers found with pattern '{layer_patterns[model_family]}'. "
            f"Check model_family or parameter names."
        )

    return [blocks[i] for i in sorted(blocks.keys())]


def compute_hvp_blockwise(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batch,
    blocks: List[List[str]],
    vector: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    真正的 Block-wise HVP 计算。

    对每个 block，仅对该 block 的参数开启 requires_grad，使得
    create_graph=True 只追踪当前 block 的二阶计算图，从而将峰值内存
    从全模型二阶图降低到单个 block 的二阶图。

    数学含义:
        对于 block b，计算 H_b θ_b，其中 H_b = ∇²_{θ_b} L
        是 loss 关于 block b 参数的 Hessian（其他参数视为常数）。
        这是 block-diagonal Hessian 近似。

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数，签名为 loss_fn(model, batch) -> scalar
        data_batch: 单个数据批次
        blocks: build_transformer_blocks 返回的 block 列表

    返回:
        HVP 结果字典 {param_name: tensor}，仅包含 block 内的参数
    """
    # 构建 name -> parameter 映射
    named_params = dict(model.named_parameters())
    all_hvp: Dict[str, torch.Tensor] = {}

    # 记录所有参数的原始 requires_grad 状态
    original_requires_grad = {
        name: p.requires_grad for name, p in named_params.items()
    }

    num_blocks = len(blocks)
    completed = False
    loss = grads = grad_vector_product = hvp_grads = None
    try:
        for block_idx, block_names in enumerate(blocks):
            block_name_set = set(block_names)
            print(f"[Block-wise HVP] Block {block_idx + 1}/{num_blocks} "
                  f"({len(block_names)} params)...")

            # 1. 只对当前 block 的参数开启 requires_grad
            for name, p in named_params.items():
                p.requires_grad_(name in block_name_set)

            # 使用 MATH backend 回退（Flash/Efficient Attention 不支持二阶导）
            with sdpa_kernel(SDPBackend.MATH):
                # 2. 前向传播（全模型，但只有当前 block 参与梯度追踪）
                model.zero_grad()
                loss = loss_fn(model, data_batch)

                # 3. 获取当前 block 的参数列表（保持顺序）
                block_params = [named_params[n] for n in block_names if n in named_params]
                block_param_names = [n for n in block_names if n in named_params]

                # 4. 计算一阶梯度（create_graph=True 仅追踪当前 block 的二阶图）
                grads = torch.autograd.grad(
                    loss,
                    block_params,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                # 5. 计算 g_b · v_b（标量）；v 默认为 θ（全权重置零场景），
                #    residual 场景应传 vector=delta 得到 H·δ
                grad_vector_product = torch.tensor(0.0, device=loss.device)
                for g, name in zip(grads, block_param_names):
                    if g is not None:
                        if vector is not None and name in vector:
                            probe = vector[name].to(g.device)
                        else:
                            probe = named_params[name].data
                        grad_vector_product = grad_vector_product + (
                            g * probe
                        ).sum()

                # 6. 二次反向传播 → H_b θ_b
                hvp_grads = torch.autograd.grad(
                    grad_vector_product,
                    block_params,
                    retain_graph=False,
                    allow_unused=True,
                )

            # 7. 保存结果，释放计算图
            for name, h in zip(block_param_names, hvp_grads):
                if h is not None:
                    all_hvp[name] = h.detach()
                else:
                    all_hvp[name] = torch.zeros_like(named_params[name].data)

            loss = grads = grad_vector_product = hvp_grads = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        completed = True
        return all_hvp
    finally:
        # The function temporarily owns requires_grad state for the whole model.
        for name, p in named_params.items():
            p.requires_grad_(original_requires_grad[name])
        if not completed:
            model.zero_grad()
        loss = grads = grad_vector_product = hvp_grads = None


def compute_hvp_blockwise_batched(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    blocks: List[List[str]],
    num_batches: int = 1,
    vector: Dict[str, torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    使用多个批次计算平均 block-wise HVP，提高估计稳定性。

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数
        data_batches: 数据批次列表
        blocks: build_transformer_blocks 返回的 block 列表
        num_batches: 使用的批次数量

    返回:
        平均 HVP 结果字典 {param_name: tensor}
    """
    hvp_sum = None
    actual_batches = min(num_batches, len(data_batches))

    for i in range(actual_batches):
        print(f"[Block-wise HVP] Batch {i + 1}/{actual_batches}...")
        hvp = compute_hvp_blockwise(model, loss_fn, data_batches[i], blocks,
                                     vector=vector)

        if hvp_sum is None:
            hvp_sum = {name: h.clone() for name, h in hvp.items()}
        else:
            for name in hvp_sum:
                if name in hvp:
                    hvp_sum[name] += hvp[name]

    if hvp_sum is None:
        return {}

    return {name: h / actual_batches for name, h in hvp_sum.items()}


def compute_importance_scores_hvp_blockwise(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    model_family: str = 'gpt2',
    num_batches: int = 1,
    alpha: float = 0.5,
    normalize: bool = False,
    grad_accumulation_batches: int = None,
) -> Dict[str, torch.Tensor]:
    """
    使用真正的 block-wise HVP 计算参数重要性得分（论文 Algorithm 1）。

    对每个 Transformer layer 独立计算 block-diagonal Hessian-Vector Product，
    内存峰值仅为单个 block 的二阶图 + 全模型前向激活，
    而非全模型的二阶计算图。

    公式: s_{ℓ,i} = |-g_{ℓ,i} · θ_{ℓ,i} + alpha · θ_{ℓ,i} · (H_b · θ_b)_{ℓ,i}|

    当 normalize=True 时，对每个参数张量做尺度归一化后再合并：
        s_{ℓ,i} = |first / μ^(1) + alpha · second / μ^(2)|
    其中 μ^(k) = mean(|term|) + ε，确保一阶与二阶项在同一量级。

    其中 H_b = ∇²_{θ_b} L 是 block-diagonal Hessian。

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数，签名为 loss_fn(model, batch) -> scalar
        data_batches: 数据批次列表
        model_family: 模型族名称（'gpt2', 'pythia', 'vit', 'bert'）
        num_batches: 用于计算 HVP 的批次数量（更多批次 = 更稳定的估计）
        alpha: 二阶项权重系数（默认 0.5，与论文一致）
        normalize: 是否对一阶/二阶项做 per-tensor 尺度归一化

    返回:
        重要性得分字典 {name: tensor}（绝对值），仅包含 Transformer block 内的参数

    示例:
        >>> def loss_fn(model, batch):
        ...     outputs = model(input_ids=batch["input_ids"], labels=batch["input_ids"])
        ...     return outputs.loss
        >>> scores = compute_importance_scores_hvp_blockwise(
        ...     model, loss_fn, batches, model_family='gpt2', num_batches=3
        ... )
    """
    # 1. 构建 block 划分
    blocks = build_transformer_blocks(model, model_family)
    print(f"[Block-wise HVP] Built {len(blocks)} blocks from {model_family} model")

    # 2. 收集模型参数权重
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone() for name, p in params.items()}

    # 3. 计算梯度（多 batch 累积，大幅提升一阶项稳定性）
    n_grad = grad_accumulation_batches if grad_accumulation_batches else len(data_batches)
    n_grad = min(n_grad, len(data_batches))
    print(f"[Block-wise HVP] Accumulating gradients over {n_grad} batches...")
    model.zero_grad()
    for _gb in range(n_grad):
        loss = loss_fn(model, data_batches[_gb])
        (loss / n_grad).backward()
        del loss

    gradients = {
        name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in params.items()
    }

    # 4. Block-wise HVP 计算
    hvp_result = compute_hvp_blockwise_batched(
        model, loss_fn, data_batches, blocks, num_batches
    )

    # 5. 计算重要性得分: s_i = |-g_i · θ_i + alpha · θ_i · (H_b · θ_b)_i|
    scores = {}
    _EPS = 1e-12
    for name in hvp_result:
        theta = weights[name]
        grad = gradients.get(name, torch.zeros_like(theta))
        hvp = hvp_result[name]

        first_order = -grad * theta
        second_order = alpha * theta * hvp

        if normalize:
            mu1 = first_order.abs().mean() + _EPS
            mu2 = second_order.abs().mean() + _EPS
            scores[name] = torch.abs(first_order / mu1 + second_order / mu2)
        else:
            scores[name] = torch.abs(first_order + second_order)

    return scores


# 保留旧接口名作为别名，向后兼容
compute_importance_scores_hvp_memory_efficient = compute_importance_scores_hvp_blockwise
