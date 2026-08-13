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

from typing import Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def _validate_real_finite_tensor(
    value: torch.Tensor,
    reference: torch.Tensor,
    label: str,
    name: str,
) -> None:
    """Validate an HVP input/output tensor against its model parameter."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} for {name!r} must be a torch.Tensor")
    if value.shape != reference.shape:
        raise ValueError(
            f"{label} shape for {name!r} must match the parameter: "
            f"{tuple(value.shape)} != {tuple(reference.shape)}"
        )
    if value.device != reference.device:
        raise ValueError(
            f"{label} device for {name!r} must match the parameter: "
            f"{value.device} != {reference.device}"
        )
    if not value.is_floating_point() or value.is_complex():
        raise TypeError(f"{label} for {name!r} must be real floating point")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{label} for {name!r} must contain only finite values")


def _validate_hvp_vector(
    params: Dict[str, torch.Tensor],
    vector: Dict[str, torch.Tensor],
) -> None:
    """Require one valid probe vector entry for every differentiated parameter."""
    if not isinstance(vector, dict):
        raise TypeError("vector must be a dictionary of named tensors")
    missing = sorted(set(params).difference(vector))
    extra = sorted(set(vector).difference(params))
    if missing or extra:
        raise ValueError(
            f"vector keys must match trainable parameters: missing={missing}, extra={extra}"
        )
    for name, parameter in params.items():
        _validate_real_finite_tensor(vector[name], parameter, "HVP vector", name)


def complete_blockwise_vector(
    model: torch.nn.Module,
    blocks: List[List[str]],
    active_vector: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Expand a partial perturbation onto all block coordinates with explicit zeros.

    ``active_vector`` names the coordinates intentionally perturbed by the
    caller.  Every other coordinate in the selected blocks is returned as a
    zero tensor, preserving Hessian cross terms without changing the probe.
    """
    if not isinstance(active_vector, Mapping):
        raise TypeError("active_vector must be a mapping of named tensors")
    named_params = dict(model.named_parameters())
    block_names = [name for block in blocks for name in block]
    unknown_blocks = sorted(set(block_names).difference(named_params))
    if unknown_blocks:
        raise ValueError(f"HVP blocks reference unknown parameters: {unknown_blocks}")
    outside_blocks = sorted(set(active_vector).difference(block_names))
    if outside_blocks:
        raise ValueError(
            f"Active HVP vector contains parameters outside the blocks: {outside_blocks}"
        )
    completed: Dict[str, torch.Tensor] = {}
    for name in block_names:
        parameter = named_params[name]
        value = active_vector.get(name)
        if value is None:
            completed[name] = torch.zeros_like(parameter)
            continue
        _validate_real_finite_tensor(value, parameter, "HVP vector", name)
        completed[name] = value
    return completed


def include_parameter_blocks(
    blocks: List[List[str]],
    required_names: Mapping[str, torch.Tensor] | set[str],
) -> List[List[str]]:
    """Append one block for required parameters outside Transformer layers."""
    required = set(required_names)
    covered = {name for block in blocks for name in block}
    outside = sorted(required.difference(covered))
    return [*blocks, outside] if outside else blocks


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


def get_flattened_scores(scores: Dict[str, torch.Tensor]) -> torch.Tensor:
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


def compute_mean_importance(scores: Dict[str, torch.Tensor]) -> float:
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
    return float(flat.mean().item())


# ============================================================
# HVP (Hessian-Vector Product) 相关函数
# ============================================================


def _validate_hvp_batch_request(data_batches: list, num_batches: int) -> int:
    """Validate a batched HVP request and return the effective batch count.

    All public batched HVP entry points use the same convention: a positive
    integer requests up to that many cached batches, and the request is capped
    by the available data.  Failing early here avoids silent empty results and
    the less useful ``data_batches[0]`` IndexError from score helpers.
    """
    if isinstance(num_batches, bool) or not isinstance(num_batches, int) or num_batches < 1:
        raise ValueError(f"num_batches must be a positive integer, got {num_batches}")
    if not data_batches:
        raise ValueError("data_batches must contain at least one batch")
    return min(num_batches, len(data_batches))


def compute_hvp(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batch: Dict[str, torch.Tensor],
    vector: Dict[str, torch.Tensor],
    gradient_accumulator: Optional[Dict[str, torch.Tensor]] = None,
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
    if not params:
        raise RuntimeError("HVP requires at least one trainable parameter")
    _validate_hvp_vector(params, vector)

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
            allow_unused=gradient_accumulator is not None,
        )

        if gradient_accumulator is not None:
            for name, gradient in zip(params, grads):
                if gradient is not None:
                    gradient_accumulator[name].add_(gradient.detach())

        # 计算 grad · vector 的标量积
        grad_vector_product = torch.tensor(0.0, device=loss.device)
        for g, name in zip(grads, params.keys()):
            if g is not None:
                grad_vector_product = grad_vector_product + (g * vector[name]).sum()

        # 第二次反向传播，计算 HVP
        if grad_vector_product.requires_grad:
            hvp_result = torch.autograd.grad(
                grad_vector_product,
                list(params.values()),
                retain_graph=False,
                allow_unused=True,
            )
        else:
            hvp_result = tuple(None for _ in params)

    # 转换为字典格式
    hvp_dict = {
        name: (torch.zeros_like(params[name]) if hvp is None else hvp.detach())
        for name, hvp in zip(params.keys(), hvp_result)
    }

    return hvp_dict


def compute_hvp_batched(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    vector: Dict[str, torch.Tensor],
    num_batches: int = 1,
    gradient_accumulator: Optional[Dict[str, torch.Tensor]] = None,
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
    actual_batches = _validate_hvp_batch_request(data_batches, num_batches)

    for i in range(actual_batches):
        batch = data_batches[i]
        hvp = compute_hvp(
            model,
            loss_fn,
            batch,
            vector,
            gradient_accumulator=gradient_accumulator,
        )

        if hvp_sum is None:
            hvp_sum = {name: h.clone() for name, h in hvp.items()}
        else:
            for name in hvp_sum:
                hvp_sum[name] += hvp[name]

    # 计算平均
    if hvp_sum is None:
        return {}

    hvp_avg = {name: h / actual_batches for name, h in hvp_sum.items()}

    return hvp_avg


_HVPGradientState = Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]


def _snapshot_hvp_model_state(
    model: torch.nn.Module,
) -> Tuple[Dict[str, bool], Dict[str, bool], _HVPGradientState]:
    """Capture mutable model state before a scoring pass."""
    modes = {name: module.training for name, module in model.named_modules()}
    requires_grad = {}
    gradients = {}
    for name, parameter in model.named_parameters():
        requires_grad[name] = parameter.requires_grad
        gradients[name] = (
            parameter.grad,
            None if parameter.grad is None else parameter.grad.detach().clone(),
        )
    return modes, requires_grad, gradients


def _restore_hvp_model_state(
    model: torch.nn.Module,
    modes: Dict[str, bool],
    requires_grad: Dict[str, bool],
    gradients: _HVPGradientState,
) -> None:
    """Restore model mode, requires-grad flags, and existing gradients."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(requires_grad[name])
        original_gradient, original_values = gradients[name]
        if original_gradient is None:
            parameter.grad = None
        else:
            assert original_values is not None
            with torch.no_grad():
                original_gradient.copy_(original_values)
            parameter.grad = original_gradient
    for name, module in model.named_modules():
        module.training = modes[name]


def _compute_importance_scores_hvp(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    num_batches: int,
    absolute: bool,
) -> Dict[str, torch.Tensor]:
    """Compute Taylor scores from matching averaged gradients and HVPs."""
    actual_batches = _validate_hvp_batch_request(data_batches, num_batches)
    params = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    weights = {name: parameter.detach().clone() for name, parameter in params.items()}
    modes, requires_grad, gradients = _snapshot_hvp_model_state(model)
    try:
        gradient_sum = {name: torch.zeros_like(parameter) for name, parameter in params.items()}
        hvp_result = compute_hvp_batched(
            model,
            loss_fn,
            data_batches,
            weights,
            num_batches=actual_batches,
            gradient_accumulator=gradient_sum,
        )
        average_gradients = {name: value / actual_batches for name, value in gradient_sum.items()}
        scores = {}
        for name, theta in weights.items():
            if name not in hvp_result:
                raise RuntimeError(f"HVP coverage is missing trainable parameter {name!r}")
            score = -average_gradients[name] * theta + 0.5 * theta * hvp_result[name]
            scores[name] = torch.abs(score) if absolute else score
        return scores
    finally:
        _restore_hvp_model_state(model, modes, requires_grad, gradients)


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
    return _compute_importance_scores_hvp(
        model,
        loss_fn,
        data_batches,
        num_batches,
        absolute=False,
    )


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
    return _compute_importance_scores_hvp(
        model,
        loss_fn,
        data_batches,
        num_batches,
        absolute=True,
    )


def build_transformer_blocks(
    model: torch.nn.Module,
    model_family: str = "gpt2",
) -> List[List[str]]:
    """
    将模型参数按 Transformer layer 分组为 block。

    每个 block 对应一个 Transformer layer，包含该层所有可训练的权重矩阵
    （Attention Q/K/V/O + MLP up/down 等）。调用方可以通过 probe vector
    把不参与扰动的坐标显式设为零；这些坐标仍留在 block 中以保留交叉项。

    参数:
        model: PyTorch 模型
        model_family: 模型族名称，用于确定参数名中的层号模式
            支持: 'gpt2', 'pythia', 'vit', 'bert'

    返回:
        List[List[str]]，每个元素是一个 block 内的参数名列表，按层号排序
    """
    import re

    layer_patterns = {
        "gpt2": r"(?:transformer\.)?h\.(\d+)\.",
        "pythia": r"gpt_neox\.layers\.(\d+)\.",
        "vit": r"(?:vit\.)?encoder\.layer\.(\d+)\.",
        "bert": r"(?:bert\.)?encoder\.layer\.(\d+)\.",
    }

    if model_family not in layer_patterns:
        raise ValueError(
            f"Unknown model_family: {model_family}. " f"Available: {list(layer_patterns.keys())}"
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
    vector: Optional[Dict[str, torch.Tensor]] = None,
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

    block_names = [name for block in blocks for name in block]
    duplicate_names = sorted({name for name in block_names if block_names.count(name) > 1})
    if duplicate_names:
        raise ValueError(f"HVP blocks contain duplicate parameter names: {duplicate_names}")
    unknown_names = sorted(set(block_names).difference(named_params))
    if unknown_names:
        raise ValueError(f"HVP blocks reference unknown parameters: {unknown_names}")
    if not block_names:
        raise ValueError("HVP blocks must contain at least one parameter")
    if vector is not None:
        expected_vector_names = {name for name in block_names}
        missing = sorted(expected_vector_names.difference(vector))
        extra = sorted(set(vector).difference(expected_vector_names))
        if missing or extra:
            raise ValueError(
                f"blockwise vector keys must match block parameters: missing={missing}, extra={extra}"
            )
        for name in block_names:
            _validate_real_finite_tensor(vector[name], named_params[name], "HVP vector", name)

    # 记录所有参数的原始 requires_grad 状态
    original_requires_grad = {name: p.requires_grad for name, p in named_params.items()}

    num_blocks = len(blocks)
    completed = False
    loss = grads = grad_vector_product = hvp_grads = None
    try:
        for block_idx, block_names in enumerate(blocks):
            block_name_set = set(block_names)
            print(
                f"[Block-wise HVP] Block {block_idx + 1}/{num_blocks} "
                f"({len(block_names)} params)..."
            )

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
                        if vector is not None:
                            probe = vector[name].to(g.device)
                        else:
                            probe = named_params[name].data
                        grad_vector_product = grad_vector_product + (g * probe).sum()

                # 6. 二次反向传播 → H_b θ_b
                if grad_vector_product.requires_grad:
                    hvp_grads = torch.autograd.grad(
                        grad_vector_product,
                        block_params,
                        retain_graph=False,
                        allow_unused=True,
                    )
                else:
                    hvp_grads = tuple(None for _ in block_params)

            # 7. 保存结果，释放计算图。None means the exact second derivative
            #    for this connected first-order coordinate is mathematically zero.
            for name, h in zip(block_param_names, hvp_grads):
                all_hvp[name] = (
                    torch.zeros_like(named_params[name].data) if h is None else h.detach()
                )

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
    vector: Optional[Dict[str, torch.Tensor]] = None,
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
    actual_batches = _validate_hvp_batch_request(data_batches, num_batches)

    for i in range(actual_batches):
        print(f"[Block-wise HVP] Batch {i + 1}/{actual_batches}...")
        hvp = compute_hvp_blockwise(model, loss_fn, data_batches[i], blocks, vector=vector)

        if hvp_sum is None:
            hvp_sum = {name: h.clone() for name, h in hvp.items()}
        else:
            if set(hvp) != set(hvp_sum):
                missing = sorted(set(hvp_sum).difference(hvp))
                extra = sorted(set(hvp).difference(hvp_sum))
                raise RuntimeError(
                    f"Blockwise HVP keys changed across batches: missing={missing}, extra={extra}"
                )
            for name in hvp_sum:
                hvp_sum[name] += hvp[name]

    if hvp_sum is None:
        return {}

    return {name: h / actual_batches for name, h in hvp_sum.items()}


def compute_importance_scores_hvp_blockwise(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    model_family: str = "gpt2",
    num_batches: int = 1,
    alpha: float = 0.5,
    normalize: bool = False,
    grad_accumulation_batches: Optional[int] = None,
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
    _validate_hvp_batch_request(data_batches, num_batches)
    if grad_accumulation_batches is not None:
        _validate_hvp_batch_request(data_batches, grad_accumulation_batches)

    # 1. Build blocks for every coordinate that the pruning perturbation can
    #    change. Embeddings and other excluded tensors must not create one huge
    #    catch-all block that defeats the memory-efficient approximation.
    blocks = build_transformer_blocks(model, model_family)
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    from dacp.pruning.pruner import filter_prunable_params

    weights = {name: parameter.data.clone() for name, parameter in params.items()}
    prunable_weights = filter_prunable_params(weights)
    if not prunable_weights:
        raise RuntimeError("Blockwise HVP scoring requires at least one prunable parameter")
    blocks = include_parameter_blocks(blocks, prunable_weights)
    print(f"[Block-wise HVP] Built {len(blocks)} blocks from {model_family} model")

    # 3. 计算梯度（多 batch 累积，大幅提升一阶项稳定性）
    n_grad = (
        len(data_batches)
        if grad_accumulation_batches is None
        else min(grad_accumulation_batches, len(data_batches))
    )
    print(f"[Block-wise HVP] Accumulating gradients over {n_grad} batches...")
    model.zero_grad()
    for _gb in range(n_grad):
        loss = loss_fn(model, data_batches[_gb])
        (loss / n_grad).backward()
        del loss

    required_names = set(prunable_weights)
    missing_gradients = sorted(
        name for name in required_names if name not in params or params[name].grad is None
    )
    if missing_gradients:
        raise RuntimeError(f"Gradient coverage is missing block parameters: {missing_gradients}")
    gradients = {name: params[name].grad.clone() for name in required_names}

    # 4. Block-wise HVP 计算
    probe = complete_blockwise_vector(model, blocks, prunable_weights)
    hvp_result = compute_hvp_blockwise_batched(
        model,
        loss_fn,
        data_batches,
        blocks,
        num_batches,
        vector=probe,
    )

    # 5. 计算重要性得分: s_i = |-g_i · θ_i + alpha · θ_i · (H_b · θ_b)_i|
    scores = {}
    _EPS = 1e-12
    for name, theta in prunable_weights.items():
        if name not in hvp_result:
            raise RuntimeError(f"HVP coverage is missing prunable parameter {name!r}")
        grad = gradients[name]
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
