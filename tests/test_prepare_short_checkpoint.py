import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.scripts.prepare_short_recovery_checkpoint import _atomic_torch_save


class TestPrepareShortCheckpoint(unittest.TestCase):
    def test_atomic_save_creates_parent_and_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "nested" / "checkpoint.pt"
            _atomic_torch_save({"weight": torch.ones(1)}, destination)

            self.assertTrue(destination.is_file())
            self.assertEqual(torch.load(destination, weights_only=True)["weight"].item(), 1.0)
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_atomic_save_preserves_existing_destination_when_save_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "checkpoint.pt"
            destination.write_bytes(b"original")

            with (
                mock.patch(
                    "experiments.scripts.prepare_short_recovery_checkpoint.torch.save",
                    side_effect=RuntimeError("write failed"),
                ),
                self.assertRaisesRegex(RuntimeError, "write failed"),
            ):
                _atomic_torch_save({}, destination)

            self.assertEqual(destination.read_bytes(), b"original")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
