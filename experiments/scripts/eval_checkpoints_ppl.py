"""
评估已有检查点的 PPL，验证微调过程是否正常。

加载 GPT-2 Medium 在 WikiText-103 上微调的 5 个检查点（step 200/400/600/800/1000），
在 validation set 上评估 perplexity，输出 PPL 随训练步数变化的趋势。

用法：
    python experiments/scripts/eval_checkpoints_ppl.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import GPT2LMHeadModel

from dacp.utils.data_loader import get_wikitext103_dataloader
from dacp.utils.paths import resolve_model_source
from experiments.lib.losses import causal_lm_loss, move_batch_to_device, perplexity_from_loss

LOCAL_MODEL_PATH = resolve_model_source("gpt2-medium", "gpt2-medium")


@torch.no_grad()
def evaluate_ppl(model, eval_loader, device, max_batches=None):
    """在 eval_loader 上计算 perplexity。"""
    if max_batches is not None and (
        isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1
    ):
        raise ValueError(f"max_batches must be None or a positive integer, got {max_batches}")
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    try:
        for i, batch in enumerate(eval_loader):
            if max_batches is not None and i >= max_batches:
                break
            batch_on_device = move_batch_to_device(batch, device)
            loss, supervised_tokens = causal_lm_loss(
                model,
                batch_on_device,
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += supervised_tokens
        if total_tokens == 0:
            raise ValueError("evaluation requires at least one supervised causal-LM token")
        avg_loss = total_loss / total_tokens
        return perplexity_from_loss(avg_loss), avg_loss
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--max_batches", type=int, default=100, help="最大评估批次数（None=全部）")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/gpt2_medium_wikitext103")
    parser.add_argument("--data_dir", type=str, default="data/wikitext103")
    args = parser.parse_args()

    # 也评估一下原始预训练模型（step 0）
    steps = [0, 200, 400, 600, 800, 1000]

    # 加载 eval 数据（只加载一次）
    print("加载 WikiText-103 validation 数据...")
    eval_loader = get_wikitext103_dataloader(
        split="validation",
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        num_workers=0,
        shuffle=False,
        local_path=args.data_dir,
    )
    print(f"Eval batches: {len(eval_loader)}, 使用前 {args.max_batches} 批")

    results = []

    for step in steps:
        print(f"\n{'='*50}")

        # 加载模型
        model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH, local_files_only=True)
        model = model.to(args.device)

        if step == 0:
            print(f"评估原始预训练模型 (step 0)...")
        else:
            ckpt_path = Path(args.ckpt_dir) / f"checkpoint_step_{step}.pt"
            if not ckpt_path.exists():
                print(f"检查点不存在: {ckpt_path}, 跳过")
                continue
            print(f"加载检查点: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            if "loss" in ckpt:
                print(f"  保存时 train loss: {ckpt['loss']:.4f}")

        # 评估
        ppl, avg_loss = evaluate_ppl(model, eval_loader, args.device, args.max_batches)
        print(f"  Step {step}: PPL = {ppl:.4f}, Avg Loss = {avg_loss:.4f}")
        results.append((step, ppl, avg_loss))

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 汇总
    print(f"\n{'='*60}")
    print("微调过程 PPL 变化趋势:")
    print(f"{'='*60}")
    print(f"{'Step':>6} | {'PPL':>10} | {'Avg Loss':>10}")
    print(f"{'-'*6}-+-{'-'*10}-+-{'-'*10}")
    for step, ppl, avg_loss in results:
        print(f"{step:>6} | {ppl:>10.4f} | {avg_loss:>10.4f}")

    if len(results) >= 2:
        first_ppl = results[0][1]
        last_ppl = results[-1][1]
        if last_ppl < first_ppl:
            print(f"\n结论: PPL 从 {first_ppl:.2f} 下降到 {last_ppl:.2f}，微调过程正常 ✓")
        else:
            print(f"\n结论: PPL 从 {first_ppl:.2f} 上升到 {last_ppl:.2f}，微调过程异常 ✗")


if __name__ == "__main__":
    main()
