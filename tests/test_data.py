import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import experiments.lib.data as data
from experiments.lib.data import cache_batches


class TestCacheBatchesContract(unittest.TestCase):
    def test_causal_lm_labels_mask_padding_without_mutating_inputs(self) -> None:
        input_ids = torch.tensor([[4, 5, 0], [6, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        labels = data.make_causal_lm_labels(input_ids, attention_mask)

        torch.testing.assert_close(
            labels,
            torch.tensor([[4, 5, -100], [6, -100, -100]]),
        )
        torch.testing.assert_close(input_ids, torch.tensor([[4, 5, 0], [6, 0, 0]]))
        self.assertNotEqual(labels.data_ptr(), input_ids.data_ptr())

    def test_causal_lm_labels_validate_tensor_contract(self) -> None:
        with self.assertRaisesRegex(TypeError, "input_ids must be a torch.Tensor"):
            data.make_causal_lm_labels([[1, 2]])
        with self.assertRaisesRegex(TypeError, "dtype torch.long"):
            data.make_causal_lm_labels(torch.ones(1, 2))
        with self.assertRaisesRegex(ValueError, "same shape"):
            data.make_causal_lm_labels(
                torch.ones(1, 2, dtype=torch.long),
                torch.ones(1, 1, dtype=torch.long),
            )
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            data.make_causal_lm_labels(
                torch.ones(1, 2, dtype=torch.long),
                torch.tensor([[1, 2]]),
            )

    def test_tokenize_alpaca_batch_masks_padding(self) -> None:
        tokenizer = mock.Mock(
            return_value={
                "input_ids": torch.tensor([[8, 9, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 0, 0]]),
            }
        )

        result = data.tokenize_alpaca_batch(
            {"instruction": ["do"], "input": [""], "output": ["done"]},
            tokenizer,
            seq_length=4,
        )

        torch.testing.assert_close(result["labels"], torch.tensor([[8, 9, -100, -100]]))
        tokenizer.assert_called_once_with(
            ["### Instruction:\ndo\n### Response:\ndone"],
            truncation=True,
            padding="max_length",
            max_length=4,
            return_tensors="pt",
        )

    def test_lm_cache_preserves_attention_mask_and_masked_labels(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "labels": torch.tensor([[1, 2, -100]]),
        }

        [cached] = cache_batches([batch], 1, "lm")

        self.assertEqual(set(cached), {"input_ids", "attention_mask", "labels"})
        for key in cached:
            torch.testing.assert_close(cached[key], batch[key])
            self.assertNotEqual(cached[key].data_ptr(), batch[key].data_ptr())

    def test_lm_cache_builds_masked_labels_when_loader_omits_them(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }

        [cached] = cache_batches([batch], 1, "lm")

        torch.testing.assert_close(cached["labels"], torch.tensor([[1, 2, -100]]))

    def test_lm_cache_masks_padding_in_loader_supplied_labels(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "labels": torch.tensor([[7, 8, 9]]),
        }

        [cached] = cache_batches([batch], 1, "lm")

        torch.testing.assert_close(cached["labels"], torch.tensor([[7, 8, -100]]))
        torch.testing.assert_close(batch["labels"], torch.tensor([[7, 8, 9]]))

    def test_lm_cache_drops_token_type_ids(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
            "token_type_ids": torch.tensor([[0, 1, 1]]),
            "labels": torch.tensor([[1, 2, -100]]),
        }

        [cached] = cache_batches([batch], 1, "lm")

        self.assertNotIn("token_type_ids", cached)

    def test_classification_cache_preserves_token_type_ids(self) -> None:
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
            "token_type_ids": torch.tensor([[0, 1, 1]]),
            "labels": torch.tensor([1]),
        }

        [cached] = cache_batches([batch], 1, "cls")

        self.assertIn("token_type_ids", cached)
        torch.testing.assert_close(cached["token_type_ids"], batch["token_type_ids"])
        self.assertNotEqual(cached["token_type_ids"].data_ptr(), batch["token_type_ids"].data_ptr())

    def test_alpaca_split_is_deterministic_and_never_empty(self) -> None:
        self.assertEqual(data._alpaca_split_sizes(2), (1, 1))
        self.assertEqual(data._alpaca_split_sizes(20), (18, 2))
        self.assertEqual(data._alpaca_split_sizes(52_000), (51_000, 1_000))
        for invalid in (True, 0, 1, 2.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "at least two"):
                data._alpaca_split_sizes(invalid)

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

    def test_alpaca_loader_discovers_json_below_data_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            alpaca_dir = root / "alpaca"
            alpaca_dir.mkdir()
            source = alpaca_dir / "alpaca_data.json"
            source.write_text(
                json.dumps([{"instruction": "i", "input": "", "output": "o"} for _ in range(20)]),
                encoding="utf-8",
            )
            tokenizer = mock.Mock(pad_token="pad", eos_token="eos")
            fake_dataset = mock.Mock()
            fake_dataset.column_names = ["instruction", "input", "output"]
            fake_dataset.__len__ = mock.Mock(return_value=20)
            fake_dataset.map.return_value = fake_dataset

            with (
                mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
                mock.patch("datasets.Dataset.from_list", return_value=fake_dataset) as from_list,
                mock.patch("datasets.load_dataset") as load_dataset,
                mock.patch("torch.utils.data.random_split", return_value=([], [])),
                mock.patch(
                    "torch.utils.data.DataLoader", side_effect=lambda dataset, **kwargs: dataset
                ),
            ):
                data._get_alpaca_loaders(
                    "pythia-410m",
                    batch_size=1,
                    seq_length=4,
                    num_workers=0,
                    data_dir=str(root),
                )

            from_list.assert_called_once()
            load_dataset.assert_not_called()

    def test_alpaca_loader_rejects_unrelated_model_before_loading_data(self) -> None:
        with (
            mock.patch("transformers.AutoTokenizer.from_pretrained") as load_tokenizer,
            mock.patch("datasets.load_dataset") as load_dataset,
        ):
            with self.assertRaisesRegex(ValueError, "supports only"):
                data._get_alpaca_loaders(
                    "gpt2-medium",
                    batch_size=1,
                    seq_length=4,
                    num_workers=0,
                    data_dir="/unused",
                )

        load_tokenizer.assert_not_called()
        load_dataset.assert_not_called()

    def test_glue_loader_uses_portable_model_source(self) -> None:
        dataset = mock.Mock()
        tokenized = mock.Mock()
        tokenized.__getitem__ = mock.Mock(
            side_effect=lambda split: [] if split in {"train", "validation_matched"} else None
        )
        dataset.map.return_value = tokenized
        tokenized.rename_column.return_value = tokenized
        with (
            mock.patch("datasets.load_dataset", return_value=dataset),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained", return_value=mock.Mock()
            ) as load,
            mock.patch(
                "dacp.utils.paths.resolve_model_source",
                return_value="/portable/bert-large-uncased",
            ) as resolve,
            mock.patch(
                "experiments.lib.data.DataLoader",
                side_effect=lambda dataset, **kwargs: dataset,
            ),
        ):
            data._get_glue_loaders(
                "bert-large",
                "mnli",
                batch_size=1,
                seq_length=4,
                num_workers=0,
                data_dir="/portable/data",
            )

        resolve.assert_called_once_with("bert-large-uncased", "bert-large-uncased")
        load.assert_called_once_with(
            "/portable/bert-large-uncased",
            cache_dir="/portable/data",
        )

    def test_glue_loader_uses_mapped_train_columns_for_token_type_ids(self) -> None:
        dataset = mock.Mock()
        mapped = mock.MagicMock()
        mapped.rename_column.return_value = mapped
        mapped.__getitem__.side_effect = {
            "train": mock.Mock(
                column_names=[
                    "premise",
                    "hypothesis",
                    "labels",
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                ]
            ),
            "validation_matched": mock.Mock(
                column_names=["input_ids", "attention_mask", "labels", "token_type_ids"]
            ),
        }.__getitem__
        dataset.map.return_value = mapped
        tokenizer = mock.Mock()

        with (
            mock.patch("datasets.load_dataset", return_value=dataset),
            mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            mock.patch("dacp.utils.paths.resolve_model_source", return_value="/portable/bert"),
            mock.patch(
                "experiments.lib.data.DataLoader",
                side_effect=lambda split, **kwargs: split,
            ),
        ):
            data._get_glue_loaders(
                "bert-large",
                "mnli",
                batch_size=1,
                seq_length=4,
                num_workers=0,
                data_dir="/portable/data",
            )

        dataset.map.assert_called_once()
        tokenizer.assert_not_called()
        mapped.set_format.assert_called_once_with(
            "torch",
            columns=["input_ids", "attention_mask", "labels", "token_type_ids"],
        )


if __name__ == "__main__":
    unittest.main()
