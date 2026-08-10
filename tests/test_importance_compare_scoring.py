import unittest

import torch

from experiments.lib.importance_compare.scoring import (
    _collect_gradients,
    _make_loss_fn,
    compute_scores_by_method,
)


class TestImportanceScoringContract(unittest.TestCase):
    def test_loss_factory_rejects_unknown_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            _make_loss_fn("unknown")

    def test_empty_methods_return_without_touching_model(self) -> None:
        self.assertEqual(
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=[],
            ),
            {},
        )

    def test_method_and_hvp_arguments_are_validated_before_model_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown importance method"):
            compute_scores_by_method(None, [], "lm", methods=["invalid"])

        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            compute_scores_by_method(None, [], "unknown", methods=["magnitude"])

        with self.assertRaisesRegex(ValueError, "Unknown hvp_mode"):
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=["magnitude"],
                hvp_mode="invalid",
            )

        with self.assertRaisesRegex(ValueError, "hvp_batches"):
            compute_scores_by_method(
                None,
                [],
                "lm",
                methods=["magnitude"],
                hvp_batches=0,
            )

    def test_gradient_collection_rejects_empty_batches_and_non_positive_count(self) -> None:
        model = torch.nn.Linear(1, 1)

        with self.assertRaisesRegex(ValueError, "cached_train"):
            _collect_gradients(model, [], "lm", 1, "cpu")

        with self.assertRaisesRegex(ValueError, "num_batches"):
            _collect_gradients(model, [{}], "lm", 0, "cpu")


if __name__ == "__main__":
    unittest.main()
