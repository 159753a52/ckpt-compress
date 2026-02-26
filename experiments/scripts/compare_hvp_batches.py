"""
对比不同批次数量的 HVP 重要性得分统计信息。
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ckpt_compress.models.gpt2 import get_gpt2_small
from src.ckpt_compress.utils.data_loader import get_wikitext103_dataloader
from src.ckpt_compress.methods.adam_prune.importance import compute_hvp_batched
import torch.nn as nn
from tqdm import tqdm


def cache_data_batches(data_loader, num_batches, device):
    """缓存数据批次。"""
    batches = []
    data_iter = iter(data_loader)
    for i in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        batches.append({
            'input_ids': batch['input_ids'].to(device),
            'labels': batch['labels'].to(device)
        })
    return batches


def compute_hvp_importance(model, data_batches, device, num_batches):
    """计算 HVP 重要性得分。"""
    model.eval()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    def loss_fn(model, batch):
        logits = model(batch['input_ids'])
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()
        return criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    weights = {name: p.data.clone() for name, p in params.items()}

    model.zero_grad()
    loss = loss_fn(model, data_batches[0])
    loss.backward()

    gradients = {
        name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in params.items()
    }

    hvp_result = compute_hvp_batched(model, loss_fn, data_batches, weights, num_batches)

    scores = {}
    for name in weights.keys():
        theta = weights[name].cpu()
        grad = gradients.get(name, torch.zeros_like(theta)).cpu()
        hvp = hvp_result.get(name, torch.zeros_like(theta)).cpu()
        first_order = -grad * theta
        second_order = 0.5 * theta * hvp
        scores[name] = first_order + second_order

    return scores


def get_statistics(scores):
    """计算统计信息。"""
    all_scores = []
    for tensor in scores.values():
        all_scores.append(tensor.flatten().numpy())
    all_scores = np.concatenate(all_scores)

    return {
        'mean': np.mean(all_scores),
        'median': np.median(all_scores),
        'std': np.std(all_scores),
        'min': np.min(all_scores),
        'max': np.max(all_scores),
        'q25': np.percentile(all_scores, 25),
        'q75': np.percentile(all_scores, 75),
        'q95': np.percentile(all_scores, 95),
        'q99': np.percentile(all_scores, 99),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("HVP 重要性得分：不同批次数量对比")
    print("=" * 80)

    # 加载模型
    print("\n加载模型...")
    model = get_gpt2_small(pretrained=False)
    checkpoint = torch.load(
        'checkpoints/gpt2_small_wikitext103_1000steps/checkpoint_step_1000_final.pt',
        map_location='cpu'
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ 模型已加载")

    # 禁用 scaled_dot_product_attention
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    # 加载数据
    print("\n加载数据...")
    data_loader = get_wikitext103_dataloader(
        split='train',
        batch_size=2,
        seq_length=256,
        num_workers=0
    )

    # 缓存 100 个批次（最多）
    print("缓存 100 个数据批次...")
    all_batches = cache_data_batches(data_loader, 100, device)

    # 测试不同的批次数量
    batch_counts = [5, 10, 20, 50, 100]

    results = {}

    for num_batches in batch_counts:
        print(f"\n{'=' * 80}")
        print(f"测试 {num_batches} 个批次...")
        print(f"{'=' * 80}")

        scores = compute_hvp_importance(model, all_batches[:num_batches], device, num_batches)
        stats = get_statistics(scores)
        results[num_batches] = stats

        print(f"\n统计信息 ({num_batches} 批次):")
        print(f"  Mean:   {stats['mean']:.6e}")
        print(f"  Median: {stats['median']:.6e}")
        print(f"  Std:    {stats['std']:.6e}")
        print(f"  Min:    {stats['min']:.6e}")
        print(f"  Max:    {stats['max']:.6e}")
        print(f"  Q25:    {stats['q25']:.6e}")
        print(f"  Q75:    {stats['q75']:.6e}")
        print(f"  Q95:    {stats['q95']:.6e}")
        print(f"  Q99:    {stats['q99']:.6e}")

    # 打印对比表格
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)
    print(f"{'批次数':<10} {'Mean':<15} {'Median':<15} {'Std':<15} {'Q95':<15}")
    print("-" * 80)
    for num_batches in batch_counts:
        stats = results[num_batches]
        print(f"{num_batches:<10} {stats['mean']:<15.6e} {stats['median']:<15.6e} "
              f"{stats['std']:<15.6e} {stats['q95']:<15.6e}")

    print("\n" + "=" * 80)
    print("结论:")
    print("  - 批次数越多，估计越稳定（标准差应该更合理）")
    print("  - 建议使用 50-100 个批次以获得可靠的重要性估计")
    print("=" * 80)


if __name__ == '__main__':
    main()
