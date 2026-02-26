"""
优化器状态工具。

提供提取和恢复优化器状态的函数，
支持从压缩检查点正确恢复训练。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Optional


def extract_optimizer_state(
    model: nn.Module,
    optimizer: optim.Optimizer
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    按参数名称索引提取优化器状态。

    对于 Adam 优化器，提取每个参数的 exp_avg 和 exp_avg_sq。

    参数:
        model: 正在优化的模型
        optimizer: 要提取状态的优化器

    返回:
        参数名称到其优化器状态的映射字典
    """
    state_dict = {}

    # 创建参数到名称的映射
    param_to_name = {
        param: name for name, param in model.named_parameters()
    }

    for param, state in optimizer.state.items():
        if param in param_to_name:
            name = param_to_name[param]

            # 提取 Adam 特定的状态
            param_state = {}
            if "exp_avg" in state:
                param_state["exp_avg"] = state["exp_avg"].clone()
            if "exp_avg_sq" in state:
                param_state["exp_avg_sq"] = state["exp_avg_sq"].clone()
            if "step" in state:
                param_state["step"] = state["step"]

            if param_state:
                state_dict[name] = param_state

    return state_dict


def restore_optimizer_state(
    model: nn.Module,
    optimizer: optim.Optimizer,
    state: Dict[str, Dict[str, torch.Tensor]]
) -> None:
    """
    从提取的状态字典恢复优化器状态。

    参数:
        model: 正在优化的模型
        optimizer: 要恢复状态的优化器
        state: 来自 extract_optimizer_state 或解压状态的状态字典
    """
    if not state:
        return

    # 创建名称到参数的映射
    name_to_param = dict(model.named_parameters())

    for name, param_state in state.items():
        if name in name_to_param:
            param = name_to_param[name]

            # 如果需要，为此参数初始化优化器状态
            if param not in optimizer.state:
                optimizer.state[param] = {}

            # 恢复状态
            if "exp_avg" in param_state:
                optimizer.state[param]["exp_avg"] = param_state["exp_avg"].clone()
            if "exp_avg_sq" in param_state:
                optimizer.state[param]["exp_avg_sq"] = param_state["exp_avg_sq"].clone()
            if "step" in param_state:
                optimizer.state[param]["step"] = param_state["step"]
            else:
                # 如果未提供 step，设置为 1（Adam 正常工作所需）
                optimizer.state[param]["step"] = torch.tensor(1.0)


def create_optimizer_with_state(
    model: nn.Module,
    state: Dict[str, Dict[str, torch.Tensor]],
    lr: float = 0.001,
    optimizer_type: str = "adam",
    **kwargs
) -> optim.Optimizer:
    """
    创建带有预加载状态的优化器。

    参数:
        model: 要优化的模型
        state: 要恢复的优化器状态
        lr: 学习率
        optimizer_type: 优化器类型（"adam"、"sgd" 等）
        **kwargs: 额外的优化器参数

    返回:
        恢复状态后的优化器
    """
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    # 如果提供了状态则恢复
    if state:
        restore_optimizer_state(model, optimizer, state)

    return optimizer
