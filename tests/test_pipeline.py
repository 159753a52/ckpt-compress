import unittest
from types import SimpleNamespace
from unittest import mock

import experiments.lib.pipeline as pipeline


class TestExperimentPipeline(unittest.TestCase):
    def test_setup_rejects_invalid_runtime_args_before_model_loading(self) -> None:
        base = dict(
            model="gpt2-small",
            dataset="wikitext2",
            checkpoint=None,
            batch_size=4,
            seq_length=128,
            data_dir="/tmp/data-root",
            num_workers=0,
            num_steps=5,
            eval_batches=2,
            device="cpu",
        )
        invalid = (
            ("batch_size", 0),
            ("seq_length", 0),
            ("num_workers", -1),
            ("num_steps", 0),
            ("eval_batches", 0),
            ("device", "mps"),
        )

        for name, value in invalid:
            args = SimpleNamespace(**{**base, name: value})
            with (
                self.subTest(name=name, value=value),
                mock.patch.object(pipeline, "load_model") as load_model,
                self.assertRaisesRegex(ValueError, name),
            ):
                pipeline.setup_model_and_data(args)
            load_model.assert_not_called()

    def test_cache_eval_false_does_not_require_or_read_eval_batches(self) -> None:
        args = SimpleNamespace(
            model="gpt2-small",
            dataset="wikitext2",
            checkpoint=None,
            batch_size=4,
            seq_length=128,
            data_dir="/tmp/data-root",
            num_workers=0,
            num_steps=1,
            eval_batches=0,
            device="cpu",
        )
        with (
            mock.patch.object(pipeline, "load_model", return_value=("model", "gpt2")),
            mock.patch.object(
                pipeline, "get_data_loaders", return_value=("train", "validation", "lm")
            ),
            mock.patch.object(pipeline, "cache_batches", return_value=["cached"]) as cache,
        ):
            result = pipeline.setup_model_and_data(args, cache_eval=False)

        self.assertIsNone(result[2])
        cache.assert_called_once_with("train", 1, "lm")

    def test_setup_forwards_data_loader_runtime_options(self) -> None:
        args = SimpleNamespace(
            model="gpt2-small",
            dataset="wikitext103",
            checkpoint=None,
            batch_size=4,
            seq_length=128,
            data_dir="/tmp/data-root",
            num_workers=3,
            num_steps=5,
            eval_batches=2,
        )

        with (
            mock.patch.object(
                pipeline,
                "load_model",
                return_value=("model", "gpt2"),
            ),
            mock.patch.object(
                pipeline,
                "get_data_loaders",
                return_value=("train-loader", "val-loader", "lm"),
            ) as get_loaders,
            mock.patch.object(
                pipeline,
                "cache_batches",
                side_effect=lambda loader, count, task: [(loader, count, task)],
            ),
        ):
            result = pipeline.setup_model_and_data(args)

        self.assertEqual(
            result[:4], ("model", [("train-loader", 5, "lm")], [("val-loader", 2, "lm")], "lm")
        )
        self.assertEqual(args.model_family, "gpt2")
        get_loaders.assert_called_once_with(
            "gpt2-small",
            "wikitext103",
            batch_size=4,
            seq_length=128,
            num_workers=3,
            data_dir="/tmp/data-root",
        )


if __name__ == "__main__":
    unittest.main()
