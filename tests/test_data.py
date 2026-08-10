import unittest

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


if __name__ == "__main__":
    unittest.main()
