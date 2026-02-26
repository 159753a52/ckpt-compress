# 离线使用 GPT-2 Medium 模型

## 问题说明

### 为什么之前需要网络？

即使模型已经下载到本地（`./data/models/models--gpt2-medium/`），HuggingFace 的 `from_pretrained()` 默认行为是：

1. ✅ 先检查本地缓存
2. ⚠️ **尝试连接网络验证版本**（即使有缓存）
3. ❌ 如果网络不可用，会报 SSL 错误

这就是为什么在运行示例时遇到网络错误的原因。

---

## 解决方案

### 已添加 `local_files_only` 参数

现在 `get_gpt2_medium()` 支持 `local_files_only` 参数，可以完全离线使用：

```python
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

# 方法 1: 默认行为（会尝试连接网络）
model = get_gpt2_medium(pretrained=True)

# 方法 2: 仅使用本地文件（不连接网络）✅ 推荐
model = get_gpt2_medium(pretrained=True, local_files_only=True)
```

---

## 使用方法

### 离线加载模型

```python
import torch
from src.ckpt_compress.models.gpt2 import get_gpt2_medium

# 加载模型（仅使用本地文件）
model = get_gpt2_medium(pretrained=True, local_files_only=True)

# 前向传播
input_ids = torch.randint(0, 50257, (2, 128))
model.eval()
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print(f"输出形状: {logits.shape}")  # torch.Size([2, 128, 50257])
```

### 验证离线功能

```bash
# 运行测试（不需要网络）
python3 << 'EOF'
from src.ckpt_compress.models.gpt2 import get_gpt2_medium
import torch

model = get_gpt2_medium(pretrained=True, local_files_only=True)
print(f"✓ 模型加载成功（离线）")

input_ids = torch.randint(0, 50257, (2, 128))
outputs = model(input_ids)
print(f"✓ 前向传播成功")
EOF
```

---

## 修改内容

### 文件: `src/ckpt_compress/models/gpt2.py`

#### 1. `GPT2MediumForExperiment.__init__()`

**之前**:
```python
def __init__(self, pretrained: bool = False):
    if pretrained:
        self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
```

**现在**:
```python
def __init__(self, pretrained: bool = False, local_files_only: bool = False):
    if pretrained:
        self.model = GPT2LMHeadModel.from_pretrained(
            'gpt2-medium',
            local_files_only=local_files_only  # ← 新增参数
        )
```

#### 2. `get_gpt2_medium()`

**之前**:
```python
def get_gpt2_medium(pretrained: bool = False):
    return GPT2MediumForExperiment(pretrained=pretrained)
```

**现在**:
```python
def get_gpt2_medium(pretrained: bool = False, local_files_only: bool = False):
    return GPT2MediumForExperiment(
        pretrained=pretrained,
        local_files_only=local_files_only  # ← 新增参数
    )
```

---

## 测试结果

```
测试 1: 使用本地文件加载模型（不连接网络）
✓ 模型加载成功（仅使用本地文件）
  参数量: 354,823,168 (354.82M)

测试 2: 前向传播
✓ 前向传播成功
  输入形状: torch.Size([2, 128])
  输出形状: torch.Size([2, 128, 50257])

✅ 所有测试通过！现在可以离线使用模型了。
```

---

## 常见问题

### Q1: 什么时候使用 `local_files_only=True`？

**A**: 在以下情况使用：
- ✅ 网络不可用或不稳定
- ✅ 想要完全离线运行
- ✅ 避免网络验证的延迟
- ✅ 确保使用特定版本的模型

### Q2: 如果本地没有模型会怎样？

**A**: 会报错：
```python
OSError: We couldn't connect to 'https://huggingface.co' to load this model,
couldn't find it in the cached files and it looks like gpt2-medium is not
the path to a directory containing a file named config.json.
```

**解决方法**: 先下载模型：
```bash
python scripts/download_models.py --model gpt2-medium
```

### Q3: 默认行为改变了吗？

**A**: 没有！为了向后兼容：
- `local_files_only` 默认为 `False`
- 不传参数时行为与之前完全相同
- 只有显式设置 `local_files_only=True` 才会离线

### Q4: 其他模型也支持吗？

**A**: 是的！同样的参数也适用于 GPT-2 Small：
```python
from src.ckpt_compress.models.gpt2 import get_gpt2_small

model = get_gpt2_small(pretrained=True, local_files_only=True)
```

---

## 上下文占用情况

根据你的问题，当前上下文使用情况：

```
当前使用: 76,739 tokens
总预算:   200,000 tokens
剩余:     123,261 tokens
占用率:   38.4%
```

还有充足的空间继续工作！

---

## 总结

### 问题
- HuggingFace 默认会尝试连接网络验证模型版本
- 即使模型已在本地，网络不可用时也会报错

### 解决方案
- ✅ 添加了 `local_files_only` 参数
- ✅ 可以完全离线使用模型
- ✅ 向后兼容，不影响现有代码

### 使用建议
```python
# 推荐：离线使用
model = get_gpt2_medium(pretrained=True, local_files_only=True)

# 或者：允许网络验证（默认）
model = get_gpt2_medium(pretrained=True)
```

---

**更新时间**: 2025-01-17
**修改文件**: `src/ckpt_compress/models/gpt2.py`
**测试状态**: ✅ 通过
