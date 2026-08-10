import math
import unittest
from typing import Mapping, Sequence
from unittest import mock

import torch

import experiments.lib.residual_calibration as residual_calibration
from experiments.lib.residual_calibration import (
    calibrate_spectral_allocation,
    calibrate_trust_region_allocation,
)


class TinyProbeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for index in range(3):
            self.register_parameter(
                f"layer{index}",
                torch.nn.Parameter(torch.ones(10)),
            )


class TestResidualCalibration(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyProbeModel()
        self.current = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        self.reference = {
            name: torch.zeros_like(value) for name, value in self.current.items()
        }
        self.layers = [[f"layer{index}"] for index in range(3)]
        self.scores = {
            f"layer{index}": torch.arange(10, dtype=torch.float32)
            for index in range(3)
        }
        self.batches = [{"batch": torch.tensor(index)} for index in range(2)]

    @staticmethod
    def analytic_losses(
        model: torch.nn.Module,
        batches: Sequence[Mapping[str, torch.Tensor]],
        device: str,
    ) -> list[float]:
        del device
        model.eval()
        pruned = [10.0 - parameter.sum().item() for parameter in model.parameters()]
        loss = sum(weight * count for weight, count in zip((1.0, 2.0, 3.0), pruned))
        return [loss] * len(batches)

    def test_trust_region_calibration_matches_analytic_marginals(self) -> None:
        with mock.patch.object(
            residual_calibration,
            "batch_loss_values",
            side_effect=self.analytic_losses,
        ):
            counts, metadata = calibrate_trust_region_allocation(
                self.model,
                self.current,
                self.reference,
                self.layers,
                self.scores,
                uniform_layer_counts=[5, 5, 5],
                target=15,
                ratio=0.5,
                max_layer_ratio=0.8,
                probe_batches=self.batches,
                selection_batches=self.batches,
                probe_radius=0.1,
                candidate_trust_radii=[0.1],
                device="cpu",
            )

        self.assertEqual(counts, [6, 5, 4])
        for actual, expected in zip(
            metadata["marginal_losses"],
            (10.0, 20.0, 30.0),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(metadata["selected_trust_radius"], 0.1)
        self.assertEqual(metadata["selected_counts"], counts)
        self.assertEqual(
            set(metadata),
            {
                "seconds",
                "probe_radius",
                "probe_batches",
                "selection_batches",
                "uniform_probe_mean_loss",
                "uniform_selection_mean_loss",
                "probes",
                "marginal_losses",
                "candidates",
                "selected_trust_radius",
                "selected_counts",
                "selection_uses_validation_data",
            },
        )
        self.assertTrue(math.isfinite(metadata["seconds"]))
        self.assertGreaterEqual(metadata["seconds"], 0.0)
        for parameter in self.model.parameters():
            self.assertTrue(torch.equal(parameter, torch.cat((torch.zeros(5), torch.ones(5)))))
            self.assertTrue(parameter.requires_grad)
        self.assertFalse(self.model.training)

    def test_spectral_calibration_matches_analytic_gradient(self) -> None:
        with mock.patch.object(
            residual_calibration,
            "batch_loss_values",
            side_effect=self.analytic_losses,
        ):
            counts_by_rank, metadata = calibrate_spectral_allocation(
                self.model,
                self.current,
                self.reference,
                self.layers,
                self.scores,
                target=15,
                ratio=0.5,
                max_layer_ratio=0.8,
                probe_batches=self.batches,
                probe_radius=0.1,
                ranks=[1, 2],
                trust_radius=0.1,
                device="cpu",
            )

        self.assertEqual(counts_by_rank, {1: [6, 5, 4], 2: [6, 5, 4]})
        self.assertEqual(metadata["direction_evaluations"], 4)
        self.assertEqual(metadata["batch_forward_evaluations"], 8)
        self.assertEqual(len(metadata["measurements"]), 2)
        self.assertEqual(set(metadata["reconstructions"]), {"1", "2"})
        self.assertEqual(metadata["returned_model_state"], "current_uncompressed")
        self.assertEqual(
            set(metadata),
            {
                "seconds",
                "mask_seconds",
                "evaluation_seconds",
                "device_cache_seconds",
                "device_cache_bytes",
                "probe_batches",
                "probe_radius",
                "trust_radius",
                "direction_evaluations",
                "batch_forward_evaluations",
                "measurements",
                "reconstructions",
                "returned_model_state",
            },
        )
        for key in (
            "seconds",
            "mask_seconds",
            "evaluation_seconds",
            "device_cache_seconds",
        ):
            self.assertTrue(math.isfinite(metadata[key]))
            self.assertGreaterEqual(metadata[key], 0.0)
        for name, parameter in self.model.named_parameters():
            self.assertTrue(torch.equal(parameter, self.current[name]))
            self.assertTrue(parameter.requires_grad)
        self.assertFalse(self.model.training)


if __name__ == "__main__":
    unittest.main()
