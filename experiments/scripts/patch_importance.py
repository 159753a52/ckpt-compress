"""
补丁脚本：修改 importance.py，为 compute_importance_scores_hvp_blockwise 增加
多 batch 梯度累积能力 (grad_accumulation_batches 参数)。

用法（在远程服务器上执行）：
    cd /root/checkpoint_compress
    python experiments/scripts/patch_importance.py

修改内容：
  - compute_importance_scores_hvp_blockwise() 新增 grad_accumulation_batches 参数
  - 一阶梯度从单 batch 改为多 batch 累积平均
  - 默认值 grad_accumulation_batches=None 表示使用所有 cached batches
"""
import shutil, sys
from pathlib import Path

TARGET = Path(__file__).parent.parent.parent / "dacp" / "tools" / "importance.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found"); sys.exit(1)

# 备份
backup = TARGET.with_suffix('.py.bak')
if not backup.exists():
    shutil.copy2(TARGET, backup)
    print(f"Backup → {backup}")

code = TARGET.read_text(encoding='utf-8')

# ============================================================
# Patch 1: 修改函数签名，增加 grad_accumulation_batches 参数
# ============================================================
OLD_SIG = """def compute_importance_scores_hvp_blockwise(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    model_family: str = 'gpt2',
    num_batches: int = 1,
    alpha: float = 0.5,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:"""

NEW_SIG = """def compute_importance_scores_hvp_blockwise(
    model: torch.nn.Module,
    loss_fn: Callable,
    data_batches: list,
    model_family: str = 'gpt2',
    num_batches: int = 1,
    alpha: float = 0.5,
    normalize: bool = False,
    grad_accumulation_batches: int = None,
) -> Dict[str, torch.Tensor]:"""

if OLD_SIG in code:
    code = code.replace(OLD_SIG, NEW_SIG, 1)
    print("Patch 1: function signature updated")
else:
    print("Patch 1: SKIPPED (signature already patched or not found)")

# ============================================================
# Patch 2: 替换单 batch 梯度计算为多 batch 累积
# ============================================================
OLD_GRAD = """    # 3. 计算梯度（一次标准反向传播，用于一阶项）
    model.zero_grad()
    loss = loss_fn(model, data_batches[0])
    loss.backward()

    gradients = {
        name: p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for name, p in params.items()
    }
    del loss"""

NEW_GRAD = """    # 3. 计算梯度（多 batch 累积，大幅提升一阶项稳定性）
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
    }"""

if OLD_GRAD in code:
    code = code.replace(OLD_GRAD, NEW_GRAD, 1)
    print("Patch 2: multi-batch gradient accumulation applied")
else:
    print("Patch 2: SKIPPED (gradient block already patched or not found)")

# ============================================================
# 写回
# ============================================================
TARGET.write_text(code, encoding='utf-8')
print(f"Patched → {TARGET}")
print("Done.")
