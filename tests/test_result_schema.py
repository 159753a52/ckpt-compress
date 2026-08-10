import json
import tempfile
import unittest
from pathlib import Path

import yaml

from experiments.lib.result_schema import (
    load_result_bundle,
    normalize_result_payload,
    primary_metric,
    result_metrics,
)
from experiments.lib.results import ResultManager
from experiments.runners.collect_results import scan_results
from experiments.scripts.aggregate_results import extract_final_metrics, load_all_results


class TestResultSchema(unittest.TestCase):
    def test_normalizes_all_supported_payloads(self) -> None:
        bare = normalize_result_payload([{"method": "bare", "loss": 3.0}])
        wrapped = normalize_result_payload({
            "results": [{"method": "wrapped", "accuracy": 0.8}],
            "config": {"model": "bert"},
        })
        summary = normalize_result_payload({
            "summary": {
                "config": {"dataset": "sst2"},
                "final": {"legacy": {"accuracy": 0.7}},
            }
        })

        self.assertEqual(bare.schema, "list")
        self.assertEqual(bare.records[0]["method"], "bare")
        self.assertEqual(wrapped.config, {"model": "bert"})
        self.assertEqual(summary.records, [{"accuracy": 0.7, "method": "legacy"}])

    def test_result_manager_roundtrip_uses_compatibility_reader(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            records = [{"method": "uniform", "metrics": {"loss": 1.5}}]
            manager.save(records, "table", "run", {"model": "tiny"})

            self.assertEqual(manager.load("table", "run"), records)
            bundle = load_result_bundle(
                Path(temp_dir) / "table" / "run" / "results.json"
            )
            self.assertEqual(bundle.config, {"model": "tiny"})

    def test_inline_config_supplements_companion_config_and_wins_on_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "run.json"
            result_path.write_text(
                json.dumps({
                    "results": [{"method": "uniform"}],
                    "config": {"model": "inline", "seed": 7},
                }),
                encoding="utf-8",
            )
            result_path.with_name("config.yaml").write_text(
                yaml.safe_dump({
                    "model": "companion",
                    "dataset": "wikitext103",
                    "seed": 1,
                }),
                encoding="utf-8",
            )

            bundle = load_result_bundle(result_path)

            self.assertEqual(
                bundle.config,
                {
                    "model": "inline",
                    "dataset": "wikitext103",
                    "seed": 7,
                },
            )

    def test_multiple_companion_configs_merge_with_json_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "run.json"
            result_path.write_text(
                json.dumps({"results": [{"method": "uniform"}]}),
                encoding="utf-8",
            )
            result_path.with_name("run_config.json").write_text(
                json.dumps({"model": "gpt2", "seed": 7}),
                encoding="utf-8",
            )
            result_path.with_name("config.yaml").write_text(
                yaml.safe_dump({"dataset": "wikitext103", "seed": 1}),
                encoding="utf-8",
            )

            bundle = load_result_bundle(result_path)

            self.assertEqual(
                bundle.config,
                {"model": "gpt2", "dataset": "wikitext103", "seed": 7},
            )

    def test_result_manager_dataframe_keeps_top_level_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            frame = manager._results_to_dataframe([
                {
                    "method": "top-level",
                    "loss": 1.5,
                    "perplexity": 4.5,
                    "accuracy": 0.0,
                },
                {
                    "method": "nested",
                    "loss": 99.0,
                    "metrics": {"loss": 2.5},
                },
            ])

            self.assertEqual(frame.loc[0, "metric_loss"], 1.5)
            self.assertEqual(frame.loc[0, "metric_perplexity"], 4.5)
            self.assertEqual(frame.loc[0, "metric_accuracy"], 0.0)
            self.assertEqual(frame.loc[1, "metric_loss"], 2.5)

    def test_primary_metric_ignores_language_model_accuracy_placeholder(self) -> None:
        self.assertEqual(
            primary_metric({"accuracy": 0, "perplexity": 12.5, "loss": 2.5}),
            ("perplexity", 12.5),
        )
        self.assertEqual(
            primary_metric({"accuracy": 0.8, "loss": 0.2}),
            ("accuracy", 0.8),
        )

    def test_primary_metric_falls_back_to_top_level_values(self) -> None:
        self.assertEqual(
            primary_metric({
                "perplexity": 12.5,
                "loss": 2.5,
                "metrics": {"seconds": 0.4},
            }),
            ("perplexity", 12.5),
        )
        self.assertEqual(
            primary_metric({
                "accuracy": 0.7,
                "metrics": {"accuracy": 0.8},
            }),
            ("accuracy", 0.8),
        )

    def test_result_metrics_centralizes_nested_override_semantics(self) -> None:
        self.assertEqual(
            result_metrics({
                "method": "example",
                "loss": 9.0,
                "perplexity": 12.5,
                "metrics": {"loss": 2.5, "seconds": 0.4},
            }),
            {"loss": 2.5, "perplexity": 12.5, "seconds": 0.4},
        )

    def test_collect_and_aggregate_share_normalized_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            managed = root / "managed" / "run"
            managed.mkdir(parents=True)
            (managed / "results.json").write_text(
                json.dumps([{"method": "bare", "loss": 2.0}]),
                encoding="utf-8",
            )
            (managed / "config.yaml").write_text(
                yaml.safe_dump({"model": "gpt2", "dataset": "wiki", "prune_ratio": 0.2}),
                encoding="utf-8",
            )
            (root / "wrapped.json").write_text(
                json.dumps({
                    "results": [
                        {"method": "wrapped", "target_ratio": 0.4, "perplexity": 12.0}
                    ],
                    "config": {"model": "pythia", "dataset": "alpaca"},
                }),
                encoding="utf-8",
            )
            (root / "summary.json").write_text(
                json.dumps({
                    "summary": {
                        "config": {"model": "bert", "dataset": "sst2"},
                        "final": {"legacy": {"accuracy": 0.75}},
                    }
                }),
                encoding="utf-8",
            )
            (root / "unrelated.json").write_text(
                json.dumps([{"distribution": "gamma", "shape": 2.0}]),
                encoding="utf-8",
            )

            collected = scan_results(root)
            self.assertEqual({record["method"] for record in collected}, {
                "bare", "wrapped", "legacy"
            })
            wrapped = next(record for record in collected if record["method"] == "wrapped")
            self.assertEqual(wrapped["prune_ratio"], 0.4)

            bundles = load_all_results(root)
            self.assertEqual(len(bundles), 3)
            entries = [entry for bundle in bundles for entry in extract_final_metrics(bundle)]
            self.assertEqual({entry["method"] for entry in entries}, {
                "bare", "wrapped", "legacy"
            })
            by_method = {entry["method"]: entry for entry in entries}
            self.assertEqual(by_method["bare"]["model"], "gpt2")
            self.assertEqual(by_method["wrapped"]["metric"], "perplexity")
            self.assertEqual(by_method["wrapped"]["prune_ratio"], 0.4)
            self.assertEqual(by_method["legacy"]["value"], 0.75)


if __name__ == "__main__":
    unittest.main()
