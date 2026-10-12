"""Exact full Gauss--Newton product for small-model probe diagnostics."""
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from experiments.lib.residual_probe_scoring import valid_shifted_logits


def exact_gn_product(model, batches, names, delta, device):
    params = dict(model.named_parameters())
    variables = [params[n] for n in names]
    probes = [delta[n].to(device) for n in names]
    out = [torch.zeros_like(d) for d in probes]
    model.eval()
    for batch in batches:
        with sdpa_kernel(SDPBackend.MATH):
            logits, _ = valid_shifted_logits(model, batch, device)
            dummy = torch.zeros_like(logits, requires_grad=True)
            pullback = torch.autograd.grad(logits, variables, dummy, create_graph=True, retain_graph=True)
            contraction = sum((g * d).sum() for g, d in zip(pullback, probes))
            tangent = torch.autograd.grad(contraction, dummy, retain_graph=True)[0].detach()
            p = logits.detach().softmax(-1)
            cotangent = p * (tangent - (p * tangent).sum(-1, keepdim=True)) / (len(batches) * logits.shape[0])
            values = torch.autograd.grad(logits, variables, cotangent)
        for target, value in zip(out, values):
            target.add_(value.detach())
        del logits, dummy, pullback, contraction, tangent, p, cotangent, values
    return dict(zip(names, out))


def self_check():
    from types import SimpleNamespace
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[.2, -.1], [.4, .3], [-.2, .5]]))
        def forward(self, ids):
            return SimpleNamespace(logits=torch.nn.functional.one_hot(ids, 2).float() @ self.weight.T)
    model = Toy()
    batch = {"input_ids": torch.tensor([[0, 1, 0]]), "labels": torch.tensor([[0, 1, 2]])}
    d = {"weight": torch.tensor([[.1, -.2], [.3, .4], [-.5, .6]])}
    exact = exact_gn_product(model, [batch], ["weight"], d, "cpu")["weight"]
    logits = model(batch["input_ids"]).logits[:, :-1].reshape(-1, 3)
    loss = torch.nn.functional.cross_entropy(logits, batch["labels"][:, 1:].flatten())
    g = torch.autograd.grad(loss, model.weight, create_graph=True)[0]
    hv = torch.autograd.grad((g * d["weight"]).sum(), model.weight)[0]
    assert torch.allclose(exact, hv, rtol=1e-5, atol=1e-6)
    print("exact GN equals the true Hessian product for linear logits")


if __name__ == "__main__":
    self_check()
