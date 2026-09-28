"""Print a digest of deterministic CPU/GPU torch computations.

Run with two interpreters that share the same torch release (for example the
cp38 install and a cp310 venv reusing its libtorch binaries); identical
digests show both produce bitwise-identical results for these operations.
"""

import hashlib
import sys

import torch


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    digest = hashlib.sha256()
    for device in ("cpu", "cuda:0"):
        generator = torch.Generator().manual_seed(1234)
        x = torch.randn(64, 256, generator=generator).to(device)
        w = torch.randn(256, 512, generator=generator).to(device).requires_grad_(True)
        target = torch.randint(0, 512, (64,), generator=generator).to(device)
        logits = torch.nn.functional.gelu(x @ w)
        loss = torch.nn.functional.cross_entropy(logits, target)
        loss.backward()
        optimizer = torch.optim.AdamW([w], lr=1e-3, weight_decay=0.01)
        optimizer.step()
        with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", enabled=device != "cpu"):
            half = (x @ w).float()
        for tensor in (logits, loss, w.grad, w, half, torch.softmax(logits, dim=-1)):
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    print(sys.version.split()[0], torch.__version__, digest.hexdigest())


if __name__ == "__main__":
    main()
