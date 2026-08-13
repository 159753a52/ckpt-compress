import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from dacp.tools.checkpoint_io import (
    get_size_breakdown,
    load_compressed_checkpoint,
    save_compressed_checkpoint,
    save_quantized_compressed_checkpoint,
)


class TestCheckpointSizeBreakdown(unittest.TestCase):
    def test_counts_packbits_padding_per_masked_tensor(self) -> None:
        state_dict = {
            "single": torch.ones(1),
            "seven": torch.ones(7),
        }
        masks = {
            "single": torch.ones(1),
            "seven": torch.ones(7),
        }

        breakdown = get_size_breakdown(state_dict, masks, use_fp16=True)

        self.assertEqual(breakdown["mask_bytes"], 2)
        self.assertEqual(breakdown["values_bytes"], 16)
        self.assertEqual(breakdown["total_compressed_bytes"], 18)
        self.assertEqual(breakdown["original_bytes"], 32)
        self.assertEqual(breakdown["sparsity"], 0)

    def test_loader_round_trips_quantized_mask_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "quantized.pt"
            torch.save(
                {
                    "weight": {
                        "shape": [2, 3],
                        "numel": 6,
                        "mask": torch.tensor([0b10100100], dtype=torch.uint8),
                        "indices": torch.tensor([0, 1, 2], dtype=torch.uint8),
                        "codebook": torch.tensor([1.25, -2.0, 3.5], dtype=torch.float16),
                    }
                },
                path,
            )

            restored = load_compressed_checkpoint(str(path))

            self.assertTrue(
                torch.equal(
                    restored["weight"],
                    torch.tensor([[1.25, 0.0, -2.0], [0.0, 0.0, 3.5]]),
                )
            )

    def test_quantized_writer_rejects_indices_that_would_overflow_uint8(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "n_clusters"):
                save_quantized_compressed_checkpoint(
                    {"weight": torch.ones(2)},
                    {"weight": torch.ones(2)},
                    str(Path(temp_dir) / "quantized.pt"),
                    n_clusters=257,
                )

    def test_checkpoint_apis_share_the_mask_contract(self) -> None:
        state_dict = {"weight": torch.ones(2, 2)}
        invalid_masks = (
            ({"weight": torch.ones(4)}, "shape"),
            ({"weight": torch.tensor([[1.0, 0.5], [0.0, 1.0]])}, "0 or 1"),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            for masks, message in invalid_masks:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(ValueError, message):
                        save_compressed_checkpoint(
                            state_dict,
                            masks,
                            str(Path(temp_dir) / "dense.pt"),
                        )
                    with self.assertRaisesRegex(ValueError, message):
                        save_quantized_compressed_checkpoint(
                            state_dict,
                            masks,
                            str(Path(temp_dir) / "quantized.pt"),
                        )
                    with self.assertRaisesRegex(ValueError, message):
                        get_size_breakdown(state_dict, masks)

    def test_size_breakdown_uses_each_tensor_dtype(self) -> None:
        state_dict = {
            "half": torch.ones(1, dtype=torch.float16),
            "double": torch.ones(1, dtype=torch.float64),
        }

        native = get_size_breakdown(state_dict, {}, use_fp16=False)
        converted = get_size_breakdown(state_dict, {}, use_fp16=True)

        self.assertEqual(native["original_bytes"], 10)
        self.assertEqual(native["values_bytes"], 10)
        self.assertEqual(converted["original_bytes"], 10)
        self.assertEqual(converted["values_bytes"], 4)

    def test_writers_reject_masks_for_unknown_state_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "checkpoint.pt")
            for writer in (
                save_compressed_checkpoint,
                save_quantized_compressed_checkpoint,
            ):
                with (
                    self.subTest(writer=writer.__name__),
                    self.assertRaisesRegex(KeyError, "absent"),
                ):
                    writer(
                        {"weight": torch.ones(1)},
                        {"missing": torch.ones(1)},
                        path,
                    )
            with self.assertRaisesRegex(KeyError, "absent"):
                get_size_breakdown(
                    {"weight": torch.ones(1)},
                    {"missing": torch.ones(1)},
                )

    def test_writer_creates_parent_and_loader_uses_tensor_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "checkpoint.pt"
            size = save_compressed_checkpoint(
                {"weight": torch.tensor([1.0, 2.0])},
                {"weight": torch.tensor([1, 0])},
                str(path),
            )

            self.assertTrue(path.is_file())
            self.assertEqual(size, path.stat().st_size)
            with mock.patch(
                "dacp.tools.checkpoint_io.torch.load",
                wraps=torch.load,
            ) as load:
                restored = load_compressed_checkpoint(str(path))
            self.assertTrue(torch.equal(restored["weight"], torch.tensor([1.0, 0.0])))
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_checkpoint_input_mappings_require_string_tensor_entries(self) -> None:
        invalid = (
            ({1: torch.ones(1)}, {}, "keys"),
            ({"weight": "bad"}, {}, "state_dict entries"),
            ({"weight": torch.ones(1)}, {"weight": "bad"}, "mask entries"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            for state, masks, message in invalid:
                with self.subTest(message=message), self.assertRaisesRegex(TypeError, message):
                    save_compressed_checkpoint(
                        state,
                        masks,
                        str(Path(temp_dir) / "checkpoint.pt"),
                    )


if __name__ == "__main__":
    unittest.main()
