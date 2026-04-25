#!/usr/bin/env python3
"""Fix garbled Chinese comments in run_fault_tolerant_training.py"""
import re

path = "experiments/scripts/run_fault_tolerant_training.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace all garbled Chinese with English equivalents
replacements = {
    '"""鍦ㄨ缁冭繃绋嬩腑鏀堕泦姊害锛堢敤浜?first-order importance scoring锛夈€?\n"""': '"""Collect gradients inline for first-order importance scoring."""',
    '"""璁粌涓€姝ワ紝杩斿洖 loss銆?\n"""': '"""Train one step and return loss."""',
    '"""缁樺埗 loss 鏇茬嚎銆?\n"""': '"""Plot loss curves."""',
    '鍥剧墖宸蹭繚瀛? ': 'Plots saved to ',
    '    # 鏇存柊 reference_weights锛堢敤浜?residual-magnitude 鍚庣画鎭㈠锛?': '    # Update reference_weights for residual-magnitude',
    '    # 涓?residual-magnitude 淇濆瓨鍒濆鏉冮噸蹇収': '    # Save initial weights for residual-magnitude',
    '    # 閲婃斁 scoring 涓棿浜х墿': '    # Release scoring intermediates',
    '    # 鍒嗙被浠诲姟鐢ㄦ洿澶?val batch 浠ユ彁楂?accuracy 鍒嗚鲸鐜?': '    # Use more val batches for classification tasks',
    '    # 閫氱敤璺緞锛歮agnitude / first-order / residual-magnitude': '    # Generic path: magnitude / first-order / residual-magnitude',
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Also strip any remaining non-ASCII from comment-only lines that look garbled
lines = text.split('\n')
cleaned = []
for line in lines:
    # If line is a comment with garbled chars (high codepoints mixed with ASCII), clean it
    stripped = line.strip()
    if stripped.startswith('#') and any('\u4e00' <= c <= '\u9fff' for c in stripped):
        # Check if it's actually garbled (contains 鍦, 鏇, etc.)
        if any(c in stripped for c in '鍦鏇璁缁冭锛堢'):
            line = line.split('#')[0] + '# (comment removed - encoding issue)'
    cleaned.append(line)

text = '\n'.join(cleaned)

with open(path, "w", encoding="utf-8", newline='\n') as f:
    f.write(text)

print("Fixed encoding issues")
