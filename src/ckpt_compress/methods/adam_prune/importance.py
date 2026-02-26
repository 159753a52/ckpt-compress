"""
AdamPrune 重要性得分计算模块。

实现两种参数重要性得分计算方式：

1. Adam 二阶矩近似（默认）：
   s_i = -g_i * θ_i + α * v_i * θ_i²
   - 使用 Adam 的 exp_avg_sq 作为 Hessian 对角线的近似
   - 计算快速，但只考虑对角元素

2. HVP（Hessian-Vector Product）精确计算：
   s_i = -g_i * θ_i + 0.5 * θ_i * (H * θ)_i
   - 使用真实的 Hessian 矩阵
   - 考虑参数之间的相互作用（非对角元素）
   - 计算较慢，但更准确

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
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable


def compute_importance_scores(
    weights: Dict[str, torch.Tensor],
    gradients: Dict[str, torch.Tensor],
    exp_avg_sq: Dict[str, torch.Tensor],
    alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    计算每个参数的重要性得分。

    公式: s_i = -g_i * θ_i + α * v_i * θ_i²

    参数:
        weights: 模型权重字典 {name: tensor}
        gradients: 梯度字典 {name: tensor}
        exp_avg_sq: Adam 二阶矩字典 {name: tensor}
        alpha: 二阶项权重系数 (默认 0.5)

    返回:
        重要性得分字典 {name: tensor}

    异常:
        KeyError: 如果 exp_avg_sq 中缺少 weights 中的键
    """
    scores = {}

    for name, weight in weights.items():
        if name not in exp_avg_sq:
            raise KeyError(f"Missing key '{name}' in exp_avg_sq")

        grad = gradients.get(name, torch.zeros_like(weight))
        v = exp_avg_sq[name]

        # s_i = -g_i * θ_i + α * v_i * θ_i² (保留符号的泰勒展开)
        first_order = -grad * weight
        second_order = alpha * v * weight ** 2

        scores[name] = first_order + second_order

    return scores


def compute_importance_scores_abs(
    weights: Dict[str, torch.Tensor],
    gradients: Dict[str, torch.Tensor],
    exp_avg_sq: Dict[str, torch.Tensor],
    alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    计算每个参数的重要性得分（绝对值版本）。

    公式: d_i = |g_i * θ_i| + α * |v_i * θ_i²|

    这个版本使用绝对值，只关注重要性的大小，不考虑符号。
    适用于只需要识别重要参数而不关心删除方向的场景。

    参数:
        weights: 模型权重字典 {name: tensor}
        gradients: 梯度字典 {name: tensor}
        exp_avg_sq: Adam 二阶矩字典 {name: tensor}
        alpha: 二阶项权重系数 (默认 0.5)

    返回:
        重要性得分字典 {name: tensor}

    异常:
        KeyError: 如果 exp_avg_sq 中缺少 weights 中的键
    """
    scores = {}

    for name, weight in weights.items():
        if name not in exp_avg_sq:
            raise KeyError(f"Missing key '{name}' in exp_avg_sq")

        grad = gradients.get(name, torch.zeros_like(weight))
        v = exp_avg_sq[name]

        # d_i = |g_i * θ_i| + α * |v_i * θ_i²|
        first_order = torch.abs(grad * weight)
        second_order = alpha * torch.abs(v * weight ** 2)

        scores[name] = first_order + second_order

    return scores


def compute_importance_scores_first_order(
    weights: Dict[str, torch.Tensor],
    gradients: Dict[str, torch.Tensor],
    exp_avg_sq: Dict[str, torch.Tensor],
    alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    计算每个参数的重要性得分（仅一阶项）。

    公式: d_i = |g_i * θ_i|

    这个版本只使用一阶梯度信息，不考虑二阶项。
    计算最简单，适合快速评估参数重要性。

    参数:
        weights: 模型权重字典 {name: tensor}
        gradients: 梯度字典 {name: tensor}
        exp_avg_sq: Adam 二阶矩字典 {name: tensor} (未使用，保持接口一致)
        alpha: 二阶项权重系数 (未使用，保持接口一致)

    返回:
        重要性得分字典 {name: tensor}
    """
    scores = {}

    for name, weight in weights.items():
        grad = gradients.get(name, torch.zeros_like(weight))

        # d_i = |g_i * θ_i| (仅一阶项)
        scores[name] = torch.abs(grad * weight)

    return scores


def compute_importance_scores_magnitude(
    weights: Dict[str, torch.Tensor],
    gradients: Optional[Dict[str, torch.Tensor]] = None,
    exp_avg_sq: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    计算每个参数的重要性得分（基于权重绝对值）。

    公式: d_i = |θ_i|

    这是最简单的重要性度量方法，仅基于权重的绝对值（magnitude）。
    不需要梯度信息，计算非常快速。

    理论依据:
    - 权重绝对值大的参数对模型输出的影响更大
    - 常用于结构化剪枝（如神经元剪枝、通道剪枝）
    - 适合作为基线方法进行对比

    优点:
    - 计算极快，不需要梯度
    - 不需要训练数据
    - 适合快速剪枝

    缺点:
    - 不考虑参数对损失的实际影响
    - 可能剪掉绝对值小但重要的参数

    参数:
        weights: 模型权重字典 {name: tensor}
        gradients: 梯度字典 {name: tensor} (未使用，保持接口一致)
        exp_avg_sq: Adam 二阶矩字典 {name: tensor} (未使用，保持接口一致)
        alpha: 二阶项权重系数 (未使用，保持接口一致)

    返回:
        重要性得分字典 {name: tensor}

    示例:
        >>> weights = {'layer1.weight': torch.randn(10, 5)}
        >>> scores = compute_importance_scores_magnitude(weights)
        >>> print(scores['layer1.weight'].shape)  # torch.Size([10, 5])
    """
    scores = {}

    for name, weight in weights.items():
        # d_i = |θ_i| (权重绝对值)
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


def compute_importance_scores_hvp_memory_efficient(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    num_batches: int = 1,
    chunk_size: int = 10,
) -> Dict[str, torch.Tensor]:
    """
    内存高效版本的 HVP 重要性计算。

    对于大模型，一次性计算所有参数的 HVP 可能导致内存不足。
    此函数分块计算，每次只处理部分参数。

    参数:
        model: PyTorch 模型
        loss_fn: 损失函数
        data_batches: 数据批次列表
        num_batches: 用于计算 HVP 的批次数量
        chunk_size: 每次处理的层数

    返回:
        重要性得分字典 {name: tensor}
    """
    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone() for name, p in params.items()}

    # 计算梯度
    model.zero_grad()
    loss = loss_fn(model, data_batches[0])
    loss.backward()

    gradients = {
        name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in params.items()
    }

    # 分块计算 HVP
    param_names = list(weights.keys())
    scores = {}

    for i in range(0, len(param_names), chunk_size):
        chunk_names = param_names[i:i + chunk_size]
        print(f"[HVP] 处理层 {i+1}-{min(i+chunk_size, len(param_names))}/{len(param_names)}...")

        # 只对当前块的参数计算 HVP
        chunk_vector = {name: weights[name] for name in chunk_names}

        # 对其他参数使用零向量
        full_vector = {name: torch.zeros_like(weights[name]) for name in weights}
        full_vector.update(chunk_vector)

        hvp_result = compute_hvp_batched(
            model, loss_fn, data_batches, full_vector, num_batches
        )

        # 计算当前块的重要性得分
        for name in chunk_names:
            theta = weights[name]
            grad = gradients.get(name, torch.zeros_like(theta))
            hvp = hvp_result.get(name, torch.zeros_like(theta))

            first_order = -grad * theta
            second_order = 0.5 * theta * hvp

            scores[name] = first_order + second_order

    return scores
