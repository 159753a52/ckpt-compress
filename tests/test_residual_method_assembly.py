import unittest
from unittest import mock

import torch

import experiments.lib.residual_method_assembly as method_assembly
from experiments.lib.residual_protocol import TAYLOR_PROBE_TRUST_METHOD


class TinyAssemblyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4))


class TestResidualMethodAssembly(unittest.TestCase):
    def setUp(self) -> None:
        self.model = TinyAssemblyModel()
        self.current_state = {"weight": torch.ones(4)}
        self.reference_state = {"weight": torch.zeros(4)}
        self.layers = [["weight"]]
        self.magnitude_scores = {"weight": torch.ones(4)}
        self.components = {
            "first_order": {"weight": torch.arange(1.0, 5.0)},
            "second_order": {"weight": torch.arange(4.0, 0.0, -1.0)},
            "taylor": {"weight": torch.tensor([0.1, 0.2, 0.4, 0.8])},
        }
        self.probe_batches = [{"input_ids": torch.tensor([[1]])}]
        self.selection_batches = [{"input_ids": torch.tensor([[2]])}]

    def config(
        self,
        spectral_ranks=(),
        quantile_smoothness_values=(),
    ) -> method_assembly.AdaptiveMethodConfig:
        return method_assembly.AdaptiveMethodConfig(
            prune_ratio=0.5,
            max_layer_ratio=0.8,
            probe_radius=0.1,
            trust_radii=(0.1,),
            spectral_ranks=spectral_ranks,
            spectral_probe_radius=0.1,
            spectral_trust_radius=0.1,
            quantile_smoothness_values=quantile_smoothness_values,
            quantile_trust_radius=0.1,
            quantile_cost_normalization="global",
            device="cpu",
        )

    def base_family(self):
        return {
            "base": {"weight": torch.tensor([False, False, True, True])},
        }, {
            "eligible_parameters": 4,
            "target_pruned": 2,
            "uniform_layer_counts": [2],
        }

    def test_probe_only_skips_score_order_and_optional_metadata(self) -> None:
        def trust_side_effect(*args):
            with torch.no_grad():
                args[0].weight.fill_(1.0)
            return [2], {"selected_trust_radius": 0.1}

        with mock.patch.object(
            method_assembly,
            "build_masks",
            side_effect=lambda *args: self.base_family(),
        ), mock.patch.object(
            method_assembly,
            "layer_score_orders",
        ) as score_orders, mock.patch.object(
            method_assembly,
            "calibrate_trust_region_allocation",
            side_effect=trust_side_effect,
        ) as trust:
            masks, metadata = method_assembly.assemble_method_family(
                self.model,
                self.current_state,
                self.reference_state,
                self.layers,
                self.magnitude_scores,
                self.components,
                self.probe_batches,
                self.selection_batches,
                self.config(),
            )

        score_orders.assert_not_called()
        self.assertEqual(tuple(masks), ("base", TAYLOR_PROBE_TRUST_METHOD))
        self.assertIs(trust.call_args.args[9], self.probe_batches)
        self.assertIs(trust.call_args.args[10], self.selection_batches)
        self.assertIsNone(trust.call_args.args[-1])
        self.assertEqual(metadata["probe_trust_layer_counts"], [2])
        self.assertEqual(metadata["whole_model_parameters"], 4)
        self.assertGreaterEqual(metadata["seconds"], 0.0)
        self.assertTrue(torch.equal(self.model.weight, torch.ones(4)))
        for key in (
            "spectral_layer_counts",
            "spectral",
            "quantile_smooth_layer_counts",
            "quantile_smooth",
            "score_order_seconds",
            "score_order_sort_device",
            "score_order_bytes",
        ):
            self.assertNotIn(key, metadata)

    def test_dynamic_methods_preserve_calibration_and_insertion_order(self) -> None:
        events = []
        orders = [torch.tensor([0, 1, 2, 3])]

        def trust_side_effect(*args):
            events.append("trust")
            self.assertIs(args[-1], orders)
            with torch.no_grad():
                args[0].weight.fill_(1.0)
            return [2], {"selected_trust_radius": 0.1}

        def spectral_side_effect(*args):
            events.append("spectral")
            self.assertIs(args[-1], orders)
            self.assertTrue(torch.equal(args[0].weight, torch.ones(4)))
            with torch.no_grad():
                args[0].weight.fill_(2.0)
            return {2: [1], 1: [3]}, {"direction_evaluations": 2}

        def quantile_side_effect(*args):
            events.append("quantile")
            self.assertIs(args[2], orders)
            self.assertTrue(torch.equal(self.model.weight, torch.full((4,), 2.0)))
            return {0.2: [2], 0.1: [2]}, {"model_forward_evaluations": 0}

        with mock.patch.object(
            method_assembly,
            "build_masks",
            side_effect=lambda *args: self.base_family(),
        ), mock.patch.object(
            method_assembly,
            "layer_score_orders",
            return_value=orders,
        ) as score_orders, mock.patch.object(
            method_assembly,
            "calibrate_trust_region_allocation",
            side_effect=trust_side_effect,
        ), mock.patch.object(
            method_assembly,
            "calibrate_spectral_allocation",
            side_effect=spectral_side_effect,
        ), mock.patch.object(
            method_assembly,
            "calibrate_quantile_smooth_allocation",
            side_effect=quantile_side_effect,
        ):
            masks, metadata = method_assembly.assemble_method_family(
                self.model,
                self.current_state,
                self.reference_state,
                self.layers,
                self.magnitude_scores,
                self.components,
                self.probe_batches,
                self.selection_batches,
                self.config(
                    spectral_ranks=(1, 2),
                    quantile_smoothness_values=(0.1, 0.2),
                ),
            )

        self.assertEqual(events, ["trust", "spectral", "quantile"])
        score_orders.assert_called_once_with(
            self.layers,
            self.components["taylor"],
            sort_device="cpu",
        )
        self.assertEqual(
            tuple(masks),
            (
                "base",
                TAYLOR_PROBE_TRUST_METHOD,
                "taylor_spectral_k2",
                "taylor_spectral_k1",
                "taylor_quantile_global_smooth_l0p2",
                "taylor_quantile_global_smooth_l0p1",
            ),
        )
        self.assertEqual(metadata["spectral_layer_counts"], {2: [1], 1: [3]})
        self.assertEqual(
            metadata["quantile_smooth_layer_counts"],
            {"0.2": [2], "0.1": [2]},
        )
        self.assertEqual(metadata["score_order_sort_device"], "cpu")
        self.assertEqual(metadata["score_order_bytes"], orders[0].nelement() * 8)
        self.assertTrue(torch.equal(self.model.weight, torch.full((4,), 2.0)))


if __name__ == "__main__":
    unittest.main()
