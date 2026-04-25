"""
补丁脚本：为 run_method_comparison.py 增加 magnitude + weibull-adaptive 方法。

用法:
    cd /root/checkpoint_compress
    python experiments/scripts/patch_methods.py
"""
import shutil, sys
from pathlib import Path

TARGET = Path(__file__).parent / "run_method_comparison.py"

if not TARGET.exists():
    print(f"ERROR: {TARGET} not found"); sys.exit(1)

backup = TARGET.with_suffix('.py.methods_bak')
if not backup.exists():
    shutil.copy2(TARGET, backup)
    print(f"Backup → {backup}")

code = TARGET.read_text(encoding='utf-8')

# ============================================================
# Patch: 在 METHODS 列表中增加 magnitude + weibull-adaptive
# ============================================================
OLD_METHODS = """METHODS = [
    {'importance': 'magnitude',          'allocation': 'uniform'},
    {'importance': 'first-order',        'allocation': 'uniform'},
    {'importance': 'first-order',        'allocation': 'weibull-adaptive'},
    {'importance': 'residual-magnitude', 'allocation': 'uniform'},
    {'importance': 'second-order-hvp',   'allocation': 'uniform'},
    {'importance': 'second-order-hvp',   'allocation': 'weibull-adaptive'},
]"""

NEW_METHODS = """METHODS = [
    {'importance': 'magnitude',          'allocation': 'uniform'},
    {'importance': 'magnitude',          'allocation': 'weibull-adaptive'},
    {'importance': 'first-order',        'allocation': 'uniform'},
    {'importance': 'first-order',        'allocation': 'weibull-adaptive'},
    {'importance': 'residual-magnitude', 'allocation': 'uniform'},
    {'importance': 'second-order-hvp',   'allocation': 'uniform'},
    {'importance': 'second-order-hvp',   'allocation': 'weibull-adaptive'},
]"""

if OLD_METHODS in code:
    code = code.replace(OLD_METHODS, NEW_METHODS, 1)
    print("Patch: magnitude+weibull-adaptive added to METHODS")
elif "{'importance': 'magnitude',          'allocation': 'weibull-adaptive'}," in code:
    print("Patch: SKIPPED (already patched)")
else:
    print("Patch: WARNING - METHODS block not found, check format")

TARGET.write_text(code, encoding='utf-8')
print(f"Patched → {TARGET}")
print("Done.")
