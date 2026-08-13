import argparse
import math
import unittest
from types import SimpleNamespace

from experiments.lib.args import (
    add_prune_ratios_arg,
    add_scoring_args,
    create_base_parser,
    parse_prune_ratios,
)


class TestSharedExperimentArgs(unittest.TestCase):
    def test_common_and_scoring_counts_fail_during_parse(self) -> None:
        invalid_options = (
            ("--batch_size", "0"),
            ("--seq_length", "-1"),
            ("--num_workers", "-1"),
            ("--num_steps", "0"),
            ("--hvp_batches", "0"),
            ("--chunk_size", "0"),
            ("--eval_batches", "0"),
            ("--alpha", "nan"),
            ("--alpha", "1.1"),
            ("--device", "mps"),
        )
        for option, value in invalid_options:
            parser = create_base_parser("test")
            add_scoring_args(parser)
            with self.subTest(option=option, value=value), self.assertRaises(SystemExit):
                parser.parse_args(
                    ["--model", "gpt2-small", "--dataset", "wikitext2", option, value]
                )

    def test_prune_ratios_are_finite_and_in_range(self) -> None:
        parser = argparse.ArgumentParser()
        add_prune_ratios_arg(parser)

        self.assertEqual(parser.parse_args([]).prune_ratios, [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(
            parser.parse_args(["--prune_ratios", "0, 0.5,1"]).prune_ratios, [0.0, 0.5, 1.0]
        )
        for value in ("", "0.5,", "-0.1", "1.1", str(math.nan)):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["--prune_ratios", value])

    def test_parse_prune_ratios_keeps_namespace_compatibility(self) -> None:
        self.assertEqual(
            parse_prune_ratios(SimpleNamespace(prune_ratios="0.25,0.75")),
            [0.25, 0.75],
        )
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            parse_prune_ratios(SimpleNamespace(prune_ratios=[0.5, float("inf")]))


if __name__ == "__main__":
    unittest.main()
