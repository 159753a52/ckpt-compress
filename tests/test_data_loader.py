import unittest
from unittest import mock

import torch

import dacp.utils.data_loader as data_loader
from dacp.utils.data_loader import GLUEDataset, WikiText2Dataset, _apply_subset


class _Dataset:
    def __len__(self) -> int:
        return 3


class _CountingTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        self.calls += 1
        return [int(value) for value in text.split()]


class _RecordingTokenizer:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, *texts, **kwargs):
        self.calls.append((texts, kwargs))
        return {
            'input_ids': torch.tensor([[1, 2]]),
            'attention_mask': torch.tensor([[1, 1]]),
        }


class TestWikiText2Dataset(unittest.TestCase):
    def test_zero_max_samples_is_empty_without_encoding(self) -> None:
        tokenizer = _CountingTokenizer()

        dataset = WikiText2Dataset(
            texts=["0 1 2 3 4 5"],
            tokenizer=tokenizer,
            seq_length=2,
            max_samples=0,
        )

        self.assertEqual(len(dataset), 0)
        self.assertEqual(tokenizer.calls, 0)

    def test_max_samples_and_sequence_length_reject_invalid_values(self) -> None:
        tokenizer = _CountingTokenizer()
        invalid_cases = (
            ("max_samples", {"max_samples": -1}),
            ("max_samples", {"max_samples": True}),
            ("seq_length", {"seq_length": 0}),
            ("seq_length", {"seq_length": False}),
        )

        for name, overrides in invalid_cases:
            with self.subTest(name=name, overrides=overrides):
                with self.assertRaisesRegex(ValueError, name):
                    WikiText2Dataset(
                        texts=[],
                        tokenizer=tokenizer,
                        **overrides,
                    )

    def test_subset_contract_rejects_negative_and_boolean_limits(self) -> None:
        for value in (-1, True):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "train_subset"):
                    _apply_subset(_Dataset(), value, "train_subset")

    def test_subset_contract_preserves_prefix_and_zero_limit(self) -> None:
        dataset = _Dataset()

        prefix = _apply_subset(dataset, 2, "train_subset")
        empty = _apply_subset(dataset, 0, "train_subset")

        self.assertEqual(prefix.indices, [0, 1])
        self.assertEqual(empty.indices, [])
        self.assertIs(_apply_subset(dataset, None, "train_subset"), dataset)


class TestGlueLoaders(unittest.TestCase):
    def test_dataset_adapter_uses_task_fields_and_label_dtype(self) -> None:
        tokenizer = _RecordingTokenizer()
        regression = GLUEDataset(
            [{'sentence1': 'left', 'sentence2': 'right', 'label': 2.5}],
            tokenizer,
            'stsb',
            max_length=16,
        )[0]
        classification = GLUEDataset(
            [{'question': 'question', 'sentence': 'answer', 'label': 1}],
            tokenizer,
            'qnli',
            max_length=8,
        )[0]

        self.assertEqual(tokenizer.calls[0][0], ('left', 'right'))
        self.assertEqual(tokenizer.calls[0][1]['max_length'], 16)
        self.assertEqual(tokenizer.calls[1][0], ('question', 'answer'))
        self.assertEqual(regression['labels'].dtype, torch.float32)
        self.assertEqual(classification['labels'].dtype, torch.int64)
        self.assertEqual(regression['input_ids'].shape, (2,))

    def test_split_builder_preserves_order_and_shared_configuration(self) -> None:
        with mock.patch.object(
            data_loader,
            'get_glue_dataloader',
            side_effect=lambda **kwargs: kwargs['split'],
        ) as loader:
            result = data_loader._get_glue_split_loaders(
                'mnli',
                (('train', 10), ('validation_matched', 3)),
                batch_size=4,
                data_dir='/tmp/glue',
                max_length=64,
                num_workers=2,
            )

        self.assertEqual(result, ('train', 'validation_matched'))
        self.assertEqual(
            [call.kwargs['subset'] for call in loader.call_args_list],
            [10, 3],
        )
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs['dataset_name'], 'mnli')
            self.assertEqual(call.kwargs['batch_size'], 4)
            self.assertEqual(call.kwargs['data_dir'], '/tmp/glue')


if __name__ == "__main__":
    unittest.main()
