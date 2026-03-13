import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class SdpaToyModel(nn.Module):
    def __init__(self, d: int = 16):
        super().__init__()
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> q/k/v: [B, H=1, T, D]
        q = self.wq(x).unsqueeze(1)
        k = self.wk(x).unsqueeze(1)
        v = self.wv(x).unsqueeze(1)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.sum()


def test_hvp_uses_math_sdpa_kernel_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from experiments.lib.importance_compare.hvp import compute_hvp_batched_full

    device = torch.device("cuda")
    model = SdpaToyModel().to(device)

    batch = {"x": torch.randn(2, 8, 16, device=device)}

    def loss_fn(m, b):
        return m(b["x"])

    vector = {n: p.detach() for n, p in model.named_parameters()}
    target_names = [n for n, _ in model.named_parameters()]

    hvp = compute_hvp_batched_full(
        model=model,
        loss_fn=loss_fn,
        data_batches=[batch],
        vector=vector,
        num_batches=1,
        target_names=target_names,
    )

    assert set(hvp.keys()) == set(target_names)
    for n, t in hvp.items():
        assert torch.isfinite(t).all(), f"Non-finite HVP for {n}"
