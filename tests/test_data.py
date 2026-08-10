import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiments.lib.data as data
from experiments.lib.data import cache_batches


class TestCacheBatchesContract(unittest.TestCase):
    def test_rejects_unknown_task_type_before_iterating_loader(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown task_type"):
            cache_batches(object(), 1, "unknown")

    def test_rejects_non_positive_batch_budget_before_iterating_loader(self) -> None:
        for value in (0, -1, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "num_batches"):
                    cache_batches(object(), value, "lm")

    def test_wikitext_local_path_resolver_supports_dataset_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "wikitext103"
            dataset_dir.mkdir()
            (dataset_dir / "train.txt").touch()
            (dataset_dir / "valid.txt").touch()

            self.assertEqual(
                data._find_wikitext_local_path(temp_dir, "wikitext103"),
                str(dataset_dir),
            )
            self.assertIsNone(data._find_wikitext_local_path(temp_dir, "wikitext2"))

    def test_data_directory_is_forwarded_to_text_loaders(self) -> None:
        cases = (
            ("wikitext103", "_get_wikitext_loaders"),
            ("sst2", "_get_glue_loaders"),
            ("alpaca", "_get_alpaca_loaders"),
        )
        for dataset_name, owner in cases:
            with self.subTest(dataset_name=dataset_name):
                with mock.patch.object(
                    data,
                    owner,
                    return_value=("train", "validation"),
                ) as loader:
                    result = data.get_data_loaders(
                        "gpt2-medium",
                        dataset_name,
                        data_dir="/tmp/data-root",
                    )

                self.assertEqual(result[-1], data.get_task_type(dataset_name))
                self.assertEqual(loader.call_args.args[-1], "/tmp/data-root")


if __name__ == "__main__":
    unittest.main()
