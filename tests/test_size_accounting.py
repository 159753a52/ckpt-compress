import math
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.scripts.size_accounting import (
    exact_residual_masks,
    load_state,
    parse_ratios,
    residuals,
    size_rows,
)


class TestSizeAccounting(unittest.TestCase):
    def test_load_state_uses_tensor_only_loader_and_nested_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "weight": torch.ones(2),
                        "counter": torch.tensor(3),
                    }
                },
                path,
            )
            state = load_state(path)
        self.assertEqual(set(state), {"weight"})

    def test_residuals_reject_key_and_shape_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "keys must match"):
            residuals({"left": torch.ones(1)}, {"right": torch.ones(1)})
        with self.assertRaisesRegex(ValueError, "shape differs"):
            residuals({"weight": torch.ones(2)}, {"weight": torch.ones(3)})

    def test_exact_masks_preserve_global_budget_across_ties_and_tensors(self) -> None:
        deltas = {"left": torch.ones(1), "right": torch.ones(3)}
        masks = exact_residual_masks(deltas, 0.5)
        self.assertEqual(sum(int((~mask).sum()) for mask in masks.values()), 2)
        self.assertEqual(sum(mask.numel() for mask in masks.values()), 4)

    def test_size_rows_report_exact_floor_budget(self) -> None:
        deltas = {"left": torch.arange(3.0), "right": torch.arange(4.0)}
        [row] = size_rows(deltas, [0.5], "int4")
        self.assertEqual(row["parameter_count"], 7)
        self.assertEqual(row["pruned_count"], math.floor(7 * 0.5))
        self.assertGreater(row["serialized_bytes"], 0)
        self.assertGreater(row["zlib_bytes"], 0)

    def test_parse_ratios_rejects_empty_duplicate_and_out_of_range_values(self) -> None:
        for value in ("", "0.5,0.5", "-0.1", "1.1", "nan"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_ratios(value)


if __name__ == "__main__":
    unittest.main()
