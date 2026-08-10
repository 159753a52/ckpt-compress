import unittest

from dacp.utils.data_loader import WikiText2Dataset


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


if __name__ == "__main__":
    unittest.main()
