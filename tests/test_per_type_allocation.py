import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from baselines.inshrinkerator.per_type_allocation import PerTypeAllocation
from baselines.inshrinkerator.per_type_search import SearchResult, save_search_result
from experiments.lib.residual_runtime import batch_hash, sha256_file
from experiments.scripts.run_inshrinkerator_like_search import (
    _build_search_payload,
    _result_name,
    _write_new_json,
)
from experiments.scripts.run_paper_fullweight_comparison import load_fullweight_manifest


class TestPerTypeAllocation(unittest.TestCase):
    @staticmethod
    def _search_result(**overrides) -> SearchResult:
        values = {
            "epsilon": 0.01,
            "best_metric": "magnitude",
            "per_type_ratios": {"attn": 0.2},
            "actual_global_ratio": 0.2,
            "baseline_metrics": {"loss": 1.0},
            "pruned_metrics": {"loss": 1.1},
            "quality_drop_pct": 10.0,
            "search_log": [],
        }
        values.update(overrides)
        return SearchResult(**values)

    def test_search_result_writer_is_strict_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(save_search_result(self._search_result(), temp_dir, "search"))
            original = path.read_text(encoding="utf-8")

            with self.assertRaises(ValueError):
                save_search_result(
                    self._search_result(baseline_metrics={"loss": float("nan")}),
                    temp_dir,
                    "search",
                )
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            self.assertEqual([item.name for item in Path(temp_dir).iterdir()], ["search.json"])

    def test_loads_validated_search_result_and_scales_ratios(self) -> None:
        payload = {
            "per_type_ratios": {"attn": 0.2, "mlp": 0.4},
            "actual_global_ratio": 0.4,
            "best_metric": "magnitude",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "search.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            allocation = PerTypeAllocation.from_json(str(path), model_family="gpt2")

        with mock.patch(
            "baselines.inshrinkerator.per_type_allocation.infer_layer_type",
            side_effect=lambda name, _score, _family: name,
        ):
            rates = allocation.allocate(
                {"attn": torch.ones(2), "mlp": torch.ones(2), "skip": torch.ones(1)},
                0.2,
            )

        self.assertEqual(rates, {"attn": 0.1, "mlp": 0.2, "skip": 0.0})
        self.assertEqual(allocation.best_metric, "magnitude")
        self.assertEqual(allocation.get_importance_method(), "magnitude")

    def test_rejects_malformed_search_results_and_ratios(self) -> None:
        invalid_payloads: tuple[object, ...] = (
            [],
            {"actual_global_ratio": 0.4},
            {"per_type_ratios": {"attn": 1.1}, "actual_global_ratio": 0.4},
            {
                "per_type_ratios": {"attn": 0.2},
                "actual_global_ratio": 0.4,
                "best_metric": "unknown",
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "search.json"
            for payload in invalid_payloads:
                with self.subTest(payload=payload):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaises((TypeError, ValueError)):
                        PerTypeAllocation.from_json(str(path))

        allocation = PerTypeAllocation({"attn": 0.2}, source_global_ratio=0.2)
        with self.assertRaisesRegex(ValueError, "global_prune_ratio"):
            allocation.allocate({"attn": torch.ones(1)}, float("nan"))

    def test_search_payload_contains_formal_provenance(self) -> None:
        batches = [{"input_ids": torch.tensor([[1, 2]]), "labels": torch.tensor([[1, 2]])}]
        result = self._search_result()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint.pt"
            checkpoint.write_bytes(b"checkpoint")
            payload = _build_search_payload(
                result,
                model="gpt2-medium",
                dataset="wikitext103",
                seed=42,
                checkpoint=checkpoint,
                checkpoint_sha256=sha256_file(checkpoint),
                source_digest="a" * 64,
                scoring_batches=batches,
                evaluation_batches=batches,
                protect_fraction=0.005,
            )

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["status"], "complete")
        self.assertEqual(payload["seed"], 42)
        self.assertEqual(payload["scoring_batch_sha256"], batch_hash(batches))
        self.assertEqual(payload["evaluation_batch_sha256"], batch_hash(batches))
        for key in (
            "model",
            "dataset",
            "checkpoint",
            "checkpoint_sha256",
            "source_digest",
            "best_metric",
            "per_type_ratios",
            "actual_global_ratio",
            "quality_drop",
        ):
            self.assertIn(key, payload)

    def test_search_writer_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "search.json"
            _write_new_json(path, {"status": "complete"})
            with self.assertRaises(FileExistsError):
                _write_new_json(path, {"status": "complete"})

    def test_formal_search_output_path_matches_fullweight_manifest(self) -> None:
        manifest = load_fullweight_manifest(
            Path(__file__).resolve().parents[1]
            / "experiments/configs/paper_fullweight_inshrinkerator.yaml"
        )
        derived = manifest.inshrinkerator_search_json.parent / _result_name(
            "gpt2-medium",
            "wikitext103",
            42,
            0.05,
            1,
        )
        self.assertEqual(derived.resolve(), manifest.inshrinkerator_search_json.resolve())


if __name__ == "__main__":
    unittest.main()
