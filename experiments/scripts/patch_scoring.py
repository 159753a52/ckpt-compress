"""
补丁脚本：修改 scoring.py，让 block-wise HVP 模式下梯度累积使用所有 cached batches，
而非仅 hvp_batches 个。

修改点：
  _compute_hvp_scores() 中的 gpu_batches 构建逻辑。
  原来: gpu_batches = cached_train[:hvp_batches]（4个batch）
  修改: gpu_batches = cached_train（所有batch，50个）

HVP 计算仍由 num_batches=hvp_batches 控制（内部 min(num_batches, len(data_batches))），
梯度累积由 patched importance.py 的 len(data_batches) 控制 → 50 batch。

用法：
    cd /root/checkpoint_compress
    python experiments/scripts/patch_scoring.py
"""
import shutil, sys
from pathlib import Path

TARGET = (
    Path(__file__).parent.parent.parent
    / "experiments" / "lib" / "importance_compare" / "scoring.py"
)

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found"); sys.exit(1)

backup = TARGET.with_suffix('.py.bak')
if not backup.exists():
    shutil.copy2(TARGET, backup)
    print(f"Backup → {backup}")

code = TARGET.read_text(encoding='utf-8')

# ============================================================
# Patch: 将仅截取 hvp_batches 个 batch 改为传入所有 cached_train
# ============================================================

# 目标行：gpu_batches 构建（_compute_hvp_scores 函数内）
# 原代码:
#   gpu_batches = []
#   for b in cached_train[:hvp_batches]:
#       gpu_batches.append({k: v.to(device) for k, v in b.items()})

OLD_GPU_BATCHES = "for b in cached_train[:hvp_batches]:"
NEW_GPU_BATCHES = "for b in cached_train:  # patched: pass ALL batches; HVP uses num_batches, grad accum uses all"

if OLD_GPU_BATCHES in code:
    code = code.replace(OLD_GPU_BATCHES, NEW_GPU_BATCHES, 1)
    print("Patch: gpu_batches now uses ALL cached_train (not just hvp_batches)")
else:
    print("Patch: SKIPPED (already patched or pattern not found)")

# ============================================================
# 写回
# ============================================================
TARGET.write_text(code, encoding='utf-8')
print(f"Patched → {TARGET}")
print("Done.")
