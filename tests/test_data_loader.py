import unittest

from dacp.utils.data_loader import WikiText2Dataset, _apply_subset


class _Dataset:
    def __len__(self) -> int:
        return 3


class _CountingTokenizer:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str) -> list[int]:
        self.calls += 1
        return [int(value) for value in text.split()]


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


if __name__ == "__main__":
    unittest.main()
