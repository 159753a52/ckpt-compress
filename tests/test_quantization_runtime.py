import unittest

import torch

from dacp.quantization import INT4Quantizer, KMeansQuantizer
from experiments.lib.quantization_runtime import (
    quantize_dequantize_tensor,
    quantize_model_parameters,
)


class TinyQuantizedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, -2.0], [3.0, -4.0]]))
        self.bias = torch.nn.Parameter(torch.tensor([1.0, -1.0]))


class TestQuantizationRuntime(unittest.TestCase):
    def test_both_quantizers_preserve_pruned_positions(self) -> None:
        weight = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        for quantizer in (KMeansQuantizer(n_clusters=2), INT4Quantizer()):
            with self.subTest(quantizer=type(quantizer).__name__):
                recovered = quantize_dequantize_tensor(weight, quantizer, mask=mask)
                self.assertTrue(torch.equal(recovered[mask == 0], torch.zeros(2)))
                self.assertTrue(torch.isfinite(recovered).all())

    def test_model_adapter_quantizes_matrices_but_not_vectors(self) -> None:
        model = TinyQuantizedModel()
        original_bias = model.bias.detach().clone()
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        quantize_model_parameters(
            model,
            {"weight": mask},
            INT4Quantizer(),
            skip_patterns=(),
        )

        self.assertTrue(torch.equal(model.weight.detach()[mask == 0], torch.zeros(2)))
        self.assertTrue(torch.equal(model.bias.detach(), original_bias))

    def test_mask_contract_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            quantize_dequantize_tensor(
                torch.ones(2, 2),
                INT4Quantizer(),
                mask=torch.ones(3),
            )


if __name__ == "__main__":
    unittest.main()
