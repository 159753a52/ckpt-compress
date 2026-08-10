import unittest

import torch

from dacp.quantization import INT4Quantizer, KMeansQuantizer


class TestQuantizers(unittest.TestCase):
    def test_kmeans_round_trip_accepts_requires_grad_input(self) -> None:
        weight = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        quantizer = KMeansQuantizer(n_clusters=3, seed=7)

        indices, metadata = quantizer.quantize(weight)
        recovered = quantizer.dequantize(indices, metadata)

        self.assertEqual(indices.shape, weight.shape)
        self.assertEqual(recovered.dtype, weight.dtype)
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertEqual(metadata["shape"], weight.shape)

    def test_kmeans_validates_shape_mask_and_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "n_clusters"):
            KMeansQuantizer(n_clusters=0)
        with self.assertRaisesRegex(ValueError, "sigma"):
            KMeansQuantizer(sigma=1.1)

        quantizer = KMeansQuantizer(n_clusters=2)
        weight = torch.ones(2, 2)
        with self.assertRaisesRegex(ValueError, "shape"):
            quantizer.quantize(weight, shape=(3,))
        with self.assertRaisesRegex(ValueError, "mask shape"):
            quantizer.quantize(weight, mask=torch.ones(3))

    def test_int4_round_trip_and_custom_range_ratio(self) -> None:
        weight = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
        quantizer = INT4Quantizer(quant_range=2)

        quantized, metadata = quantizer.quantize(weight)
        recovered = quantizer.dequantize(quantized, metadata)

        self.assertEqual(recovered.shape, weight.shape)
        self.assertEqual(recovered.dtype, weight.dtype)
        self.assertTrue(torch.isfinite(recovered).all())
        self.assertEqual(quantizer.get_compression_ratio(), 16.0)

        with self.assertRaisesRegex(ValueError, "quant_range"):
            INT4Quantizer(quant_range=1)
        with self.assertRaisesRegex(ValueError, "quant_range"):
            INT4Quantizer(quant_range=129)
        with self.assertRaisesRegex(ValueError, "empty"):
            quantizer.quantize(torch.empty(0))

    def test_quantizers_reject_nonfinite_inputs(self) -> None:
        for quantizer in (INT4Quantizer(), KMeansQuantizer(n_clusters=2)):
            with self.subTest(quantizer=type(quantizer).__name__):
                with self.assertRaisesRegex(ValueError, "finite"):
                    quantizer.quantize(torch.tensor([1.0, float("nan")]))

    def test_shared_relative_error_handles_scalars_and_validates_shapes(self) -> None:
        original = torch.tensor([2.0])
        recovered = torch.tensor([1.5])

        for quantizer in (INT4Quantizer(), KMeansQuantizer(n_clusters=2)):
            with self.subTest(quantizer=type(quantizer).__name__):
                error = quantizer.compute_quantization_error(original, recovered)
                self.assertTrue(torch.isfinite(torch.tensor(error)))
                with self.assertRaisesRegex(ValueError, "shapes"):
                    quantizer.compute_quantization_error(
                        torch.ones(2),
                        torch.ones(1),
                    )


if __name__ == "__main__":
    unittest.main()
