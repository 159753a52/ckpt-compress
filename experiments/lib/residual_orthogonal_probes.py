"""Uniform orthogonal Hadamard probes of the full probability-factor output.

The probability weights are detached. Unique sampled Hadamard columns are
orthogonal before the parameter pullback; parameter gradients need not be.
The prefix mean is unbiased for the full Gauss--Newton product.
"""
import math
import random
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from experiments.lib.residual_probe_scoring import valid_shifted_logits
from experiments.lib.residual_runtime import lm_loss


def hadamard_signs(indices, frequency):
    bits = indices & frequency
    for shift in (32, 16, 8, 4, 2, 1):
        bits = bits ^ (bits >> shift)
    return 1 - 2 * (bits & 1).to(torch.float32)


def orthogonal_scores(model, batches, names, delta, device, rounds, seed=42,
                      report_rounds=(), on_prefix=None, on_direction=None, batch_weights=None,
                      on_batch_direction=None):
    batch_weights = [1 / len(batches)] * len(batches) if batch_weights is None else list(batch_weights)
    if len(batch_weights) != len(batches) or min(batch_weights) <= 0 or not math.isclose(sum(batch_weights), 1., rel_tol=1e-6):
        raise ValueError("batch weights must be positive and sum to one")
    params = dict(model.named_parameters())
    variables = [params[n] for n in names]
    probes = [delta[n].to(device, torch.float32) for n in names]
    gradient = [torch.zeros_like(x) for x in probes]
    sizes = []
    model.eval()
    for batch, batch_weight in zip(batches, batch_weights):
        with sdpa_kernel(SDPBackend.MATH):
            loss = lm_loss(model, batch, device)
            values = torch.autograd.grad(loss, variables)
        for target, value in zip(gradient, values):
            target.add_(value.detach(), alpha=batch_weight)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            logits, _ = valid_shifted_logits(model, batch, device)
        sizes.append(logits.numel())
        del loss, values, logits
    dimension = 1 << (sum(sizes) - 1).bit_length()
    if not 1 <= rounds <= dimension:
        raise ValueError("probe count exceeds the padded output dimension")
    frequencies = random.Random(seed).sample(range(dimension), rounds)
    generator = torch.Generator(device=device).manual_seed(seed)
    diagonal = (2 * torch.randint(0, 2, (dimension,), generator=generator, device=device, dtype=torch.int8) - 1)
    accumulated = [torch.zeros_like(x) for x in probes]
    records = []
    started = time.perf_counter()
    for index, frequency in enumerate(frequencies):
        direction = [torch.zeros_like(x) for x in probes]
        offset = 0
        for batch_index, (batch, size, batch_weight) in enumerate(zip(batches, sizes, batch_weights)):
            with sdpa_kernel(SDPBackend.MATH):
                logits, _ = valid_shifted_logits(model, batch, device)
                logp = logits.log_softmax(-1)
                positions = logits.shape[0]
                signs = hadamard_signs(torch.arange(offset, offset + size, device=device), frequency)
                signs.mul_(diagonal[offset:offset + size])
                weights = logp.detach().exp().sqrt() * signs.reshape(logits.shape) * math.sqrt(batch_weight / positions)
                scalar = -(weights * logp).sum()
                values = torch.autograd.grad(scalar, variables)
            if on_batch_direction is not None:
                on_batch_direction(index + 1, batch_index, dict(zip(names, values)))
            for target, value in zip(direction, values):
                target.add_(value.detach())
            offset += size
            del logits, logp, weights, signs, scalar, values
        alpha = torch.zeros((), device=device, dtype=torch.float64)
        for value, d in zip(direction, probes):
            alpha.add_((value.double() * d.double()).sum())
        for target, value in zip(accumulated, direction):
            target.add_(value, alpha=float(alpha))
        count = index + 1
        rec = {"rounds": count, "frequency": frequency, "padded_output_dimension": dimension,
               "expected_mse_factor": (dimension - count) / (count * (dimension - 1)),
               "probability_weighting": "all categories; sqrt(p) detached", "cumulative_seconds": time.perf_counter() - started}
        records.append(rec)
        if on_direction is not None:
            on_direction(count, dict(zip(names, direction)), dict(zip(names, gradient)), rec)
        if count in report_rounds and on_prefix is not None:
            mean = {n: u / count for n, u in zip(names, accumulated)}
            # The curvature term is d_i * (F d)_i.
            scores = {n: (-g * d + .5 * d * mean[n]).abs().cpu() for n, g, d in zip(names, gradient, probes)}
            on_prefix(count, scores, mean, rec)
            del scores, mean
        del direction
    scores = {n: (-g * d + .5 * d * u / rounds).abs().cpu()
              for n, g, d, u in zip(names, gradient, probes, accumulated)}
    return scores, {"rounds": rounds, "frequencies": frequencies, "padded_output_dimension": dimension,
                    "raw_output_dimension": sum(sizes), "records": records,
                    "operator": "full Gauss-Newton", "orthogonality_space": "padded probability-factor output space",
                    "parameter_gradients_orthogonal": False}


def self_check():
    from types import SimpleNamespace
    from experiments.lib.residual_probe_reference import exact_gn_product
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[.2, -.1], [.4, .3], [-.2, .5]]))
        def forward(self, ids):
            return SimpleNamespace(logits=torch.nn.functional.one_hot(ids, 2).float() @ self.weight.T)
    indices = torch.arange(8)
    matrix = torch.stack([hadamard_signs(indices, f) for f in range(8)])
    assert torch.equal(matrix @ matrix.T, 8 * torch.eye(8))
    model = Toy()
    batch = {"input_ids": torch.tensor([[0, 1, 0]]), "labels": torch.tensor([[0, 1, 2]])}
    d = {"weight": torch.tensor([[.1, -.2], [.3, .4], [-.5, .6]])}
    truth = exact_gn_product(model, [batch], ["weight"], d, "cpu")
    captured = []
    quadratic = []
    def capture_direction(count, values, gradient, metadata):
        quadratic.append(.5 * float((values["weight"] * d["weight"]).sum()) ** 2)
    orthogonal_scores(model, [batch], ["weight"], d, "cpu", 8, report_rounds=(8,),
                      on_prefix=lambda m, scores, u, meta: captured.append(u["weight"].clone()),
                      on_direction=capture_direction)
    assert torch.allclose(captured[0], truth["weight"], atol=1e-6, rtol=1e-5)
    assert math.isclose(sum(quadratic) / 8, .5 * float((truth["weight"] * d["weight"]).sum()), rel_tol=1e-5, abs_tol=1e-6)
    second = {"input_ids": torch.tensor([[1, 0, 1]]), "labels": torch.tensor([[1, 2, -100]])}
    weighted = exact_gn_product(model, [batch, second], ["weight"], d, "cpu", batch_weights=[.25, .75])
    captured.clear()
    separated = []
    orthogonal_scores(model, [batch, second], ["weight"], d, "cpu", 16, batch_weights=[.25, .75],
                      report_rounds=(16,), on_prefix=lambda m, scores, u, meta: captured.append(u["weight"].clone()),
                      on_batch_direction=lambda m, b, values: separated.append(.5 * float((values["weight"] * d["weight"]).sum()) ** 2))
    assert torch.allclose(captured[0], weighted["weight"], atol=1e-6, rtol=1e-5)
    assert math.isclose(sum(separated)/16, .5*float((weighted["weight"]*d["weight"]).sum()), abs_tol=1e-6, rel_tol=1e-5)
    print("Hadamard columns are orthogonal; complete basis recovers exact GN")


if __name__ == "__main__":
    self_check()
