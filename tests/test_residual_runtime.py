import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.lib.residual_recovery import (
    configure_hf_offline,
    empty_device_cache,
    peak_memory_bytes,
    reset_peak_memory,
    synchronize_device,
)


ROOT = Path(__file__).resolve().parents[1]


class TestResidualRuntime(unittest.TestCase):
    def test_import_has_no_hf_or_model_stack_side_effects(self) -> None:
        environment = os.environ.copy()
        environment.pop("HF_DATASETS_OFFLINE", None)
        environment.pop("TRANSFORMERS_OFFLINE", None)
        script = """
import json
import os
import sys
import experiments.lib.residual_recovery
print(json.dumps({
    "hf_datasets_offline": os.environ.get("HF_DATASETS_OFFLINE"),
    "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    "gpt2_loaded": "dacp.models.gpt2" in sys.modules,
    "importance_loaded": "dacp.tools.importance" in sys.modules,
    "data_loader_loaded": "dacp.utils.data_loader" in sys.modules,
    "transformers_loaded": "transformers" in sys.modules,
    "datasets_loaded": "datasets" in sys.modules,
}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            json.loads(completed.stdout),
            {
                "hf_datasets_offline": None,
                "transformers_offline": None,
                "gpt2_loaded": False,
                "importance_loaded": False,
                "data_loader_loaded": False,
                "transformers_loaded": False,
                "datasets_loaded": False,
            },
        )

    def test_configure_hf_offline_sets_defaults_without_overwriting(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            configure_hf_offline()
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

        explicit = {
            "HF_DATASETS_OFFLINE": "custom-datasets",
            "TRANSFORMERS_OFFLINE": "custom-transformers",
        }
        with mock.patch.dict(os.environ, explicit, clear=True):
            configure_hf_offline()
            self.assertEqual(os.environ["HF_DATASETS_OFFLINE"], "custom-datasets")
            self.assertEqual(
                os.environ["TRANSFORMERS_OFFLINE"], "custom-transformers"
            )

    @mock.patch.object(torch.cuda, "synchronize")
    @mock.patch.object(torch.cuda, "max_memory_allocated")
    @mock.patch.object(torch.cuda, "reset_peak_memory_stats")
    @mock.patch.object(torch.cuda, "empty_cache")
    def test_cpu_runtime_helpers_skip_cuda(
        self,
        empty_cache_mock: mock.Mock,
        reset_peak_mock: mock.Mock,
        max_memory_mock: mock.Mock,
        synchronize_mock: mock.Mock,
    ) -> None:
        empty_device_cache("cpu")
        reset_peak_memory("cpu")
        self.assertEqual(peak_memory_bytes("cpu"), 0)
        synchronize_device("cpu")

        empty_cache_mock.assert_not_called()
        reset_peak_mock.assert_not_called()
        max_memory_mock.assert_not_called()
        synchronize_mock.assert_not_called()

    @mock.patch.object(torch.cuda, "synchronize")
    @mock.patch.object(torch.cuda, "max_memory_allocated", return_value=123)
    @mock.patch.object(torch.cuda, "reset_peak_memory_stats")
    @mock.patch.object(torch.cuda, "empty_cache")
    def test_cuda_runtime_helpers_delegate_to_torch(
        self,
        empty_cache_mock: mock.Mock,
        reset_peak_mock: mock.Mock,
        max_memory_mock: mock.Mock,
        synchronize_mock: mock.Mock,
    ) -> None:
        empty_device_cache("cuda")
        reset_peak_memory("cuda")
        self.assertEqual(peak_memory_bytes("cuda"), 123)
        synchronize_device("cuda:0")

        empty_cache_mock.assert_called_once_with()
        reset_peak_mock.assert_called_once_with()
        max_memory_mock.assert_called_once_with()
        synchronize_mock.assert_called_once_with(torch.device("cuda:0"))


if __name__ == "__main__":
    unittest.main()
