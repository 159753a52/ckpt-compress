import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dacp.utils import paths


class TestPortablePaths(unittest.TestCase):
    def test_data_and_model_roots_honor_explicit_and_environment_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            model_dir = Path(temp_dir) / "models"
            with mock.patch.dict(
                os.environ,
                {
                    paths.DATA_ROOT_ENV: str(data_dir),
                    paths.MODEL_ROOT_ENV: str(model_dir),
                },
            ):
                self.assertEqual(paths.data_root(), data_dir.resolve())
                self.assertEqual(paths.model_root(), model_dir.resolve())

            explicit = Path(temp_dir) / "explicit"
            self.assertEqual(paths.data_root(explicit), explicit.resolve())
            self.assertEqual(paths.model_root(explicit), explicit.resolve())

    def test_model_source_prefers_existing_snapshot_and_falls_back_to_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(
                paths.resolve_model_source("gpt2-medium", "gpt2", root=root),
                "gpt2",
            )
            snapshot = root / "gpt2-medium"
            snapshot.mkdir()
            self.assertEqual(
                paths.resolve_model_source("gpt2-medium", "gpt2", root=root),
                str(snapshot.resolve()),
            )

    def test_data_file_resolution_stays_under_configured_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            expected = Path(temp_dir).resolve() / "alpaca" / "alpaca_data.json"
            self.assertEqual(
                paths.resolve_data_file("alpaca/alpaca_data.json", root=temp_dir),
                expected,
            )


if __name__ == "__main__":
    unittest.main()
