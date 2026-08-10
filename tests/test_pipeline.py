import unittest
from types import SimpleNamespace
from unittest import mock

import experiments.lib.pipeline as pipeline


class TestExperimentPipeline(unittest.TestCase):
    def test_setup_forwards_data_loader_runtime_options(self) -> None:
        args = SimpleNamespace(
            model='gpt2-small',
            dataset='wikitext103',
            checkpoint=None,
            batch_size=4,
            seq_length=128,
            data_dir='/tmp/data-root',
            num_workers=3,
            num_steps=5,
            eval_batches=2,
        )

        with mock.patch.object(
            pipeline,
            'load_model',
            return_value=('model', 'gpt2'),
        ), mock.patch.object(
            pipeline,
            'get_data_loaders',
            return_value=('train-loader', 'val-loader', 'lm'),
        ) as get_loaders, mock.patch.object(
            pipeline,
            'cache_batches',
            side_effect=lambda loader, count, task: [(loader, count, task)],
        ):
            result = pipeline.setup_model_and_data(args)

        self.assertEqual(result[:4], ('model', [('train-loader', 5, 'lm')],
                                      [('val-loader', 2, 'lm')], 'lm'))
        self.assertEqual(args.model_family, 'gpt2')
        get_loaders.assert_called_once_with(
            'gpt2-small',
            'wikitext103',
            batch_size=4,
            seq_length=128,
            num_workers=3,
            data_dir='/tmp/data-root',
        )


if __name__ == '__main__':
    unittest.main()
