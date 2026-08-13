import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from experiments.lib.result_schema import (
    load_result_bundle,
    normalize_result_payload,
    primary_metric,
    result_metrics,
)
from experiments.lib.results import ResultManager
from experiments.lib.results import save_results as save_legacy_results
from experiments.scripts.aggregate_results import (
    ResultScanError,
    extract_final_metrics,
    load_all_results,
    scan_results,
)


class TestResultSchema(unittest.TestCase):
    def test_normalizes_all_supported_payloads(self) -> None:
        bare = normalize_result_payload([{"method": "bare", "loss": 3.0}])
        wrapped = normalize_result_payload(
            {
                "results": [{"method": "wrapped", "accuracy": 0.8}],
                "config": {"model": "bert"},
            }
        )
        summary = normalize_result_payload(
            {
                "summary": {
                    "config": {"dataset": "sst2"},
                    "final": {"legacy": {"accuracy": 0.7}},
                }
            }
        )

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
            bundle = load_result_bundle(Path(temp_dir) / "table" / "run" / "results.json")
            self.assertEqual(bundle.config, {"model": "tiny"})

    def test_result_manager_rejects_non_finite_evidence_before_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            result_path = Path(temp_dir) / "table" / "run" / "results.json"
            manager.save([{"method": "valid", "loss": 1.0}], "table", "run")
            original = result_path.read_text(encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "non-finite"):
                manager.save([{"method": "invalid", "loss": float("nan")}], "table", "run")

            self.assertEqual(result_path.read_text(encoding="utf-8"), original)

    def test_result_manager_refuses_to_overwrite_an_existing_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            manager.save([{"method": "first", "loss": 1.0}], "table", "run")
            result_path = Path(temp_dir) / "table" / "run" / "results.json"
            original = result_path.read_text(encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                manager.save([{"method": "second", "loss": 2.0}], "table", "run")

            self.assertEqual(result_path.read_text(encoding="utf-8"), original)

    def test_result_serializer_rejects_unknown_objects_instead_of_stringifying(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            with self.assertRaisesRegex(TypeError, "unsupported result value type"):
                manager.save([{"method": "bad", "payload": object()}], "table", "run")
            self.assertFalse((Path(temp_dir) / "table" / "run").exists())

    def test_result_manager_validates_config_before_creating_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            with self.assertRaisesRegex(ValueError, "config.*non-finite"):
                manager.save(
                    [{"method": "valid", "loss": 1.0}],
                    "table",
                    "run",
                    {"learning_rate": float("inf")},
                )
            self.assertFalse((Path(temp_dir) / "table" / "run").exists())

    def test_result_manager_rejects_unsafe_path_components(self) -> None:
        unsafe = ("", " ", "..", "../escape", "nested/run", r"nested\run")
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            for value in unsafe:
                with self.subTest(experiment_name=value):
                    with self.assertRaises(ValueError):
                        manager.save([], value, "run")
                with self.subTest(run_name=value):
                    with self.assertRaises(ValueError):
                        manager.save([], "table", value)
            with self.assertRaises(ValueError):
                save_legacy_results([], temp_dir, "../escape")

    def test_result_manager_csv_failure_leaves_no_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ResultManager(temp_dir)
            with mock.patch.object(manager, "_save_csv", side_effect=RuntimeError("csv")):
                with self.assertRaisesRegex(RuntimeError, "csv"):
                    manager.save([{"loss": 1.0}], "table", "run")
            experiment_path = Path(temp_dir) / "table"
            self.assertFalse((experiment_path / "run").exists())
            self.assertEqual(list(experiment_path.iterdir()), [])

    def test_legacy_csv_failure_publishes_no_evidence_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("experiments.lib.results._get_pd") as get_pd:
                get_pd.return_value.DataFrame.return_value.to_csv.side_effect = RuntimeError("csv")
                with self.assertRaisesRegex(RuntimeError, "csv"):
                    save_legacy_results([{"loss": 1.0}], temp_dir, "table")
            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_legacy_result_writer_avoids_same_second_filename_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            save_legacy_results([{"method": "first", "loss": 1.0}], temp_dir, "table")
            save_legacy_results([{"method": "second", "loss": 2.0}], temp_dir, "table")

            payloads = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in Path(temp_dir).glob("*/results.json")
            ]
            self.assertEqual(len(payloads), 2)
            self.assertEqual(
                {payload["results"][0]["method"] for payload in payloads},
                {"first", "second"},
            )

    def test_compatibility_reader_rejects_nonfinite_results_and_configs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result_path = root / "run.json"
            result_path.write_text(
                '{"results":[{"method":"bad","loss":NaN}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-standard numeric constant NaN"):
                load_result_bundle(result_path)

            result_path.write_text(
                json.dumps({"results": [{"method": "valid", "loss": 1.0}]}),
                encoding="utf-8",
            )
            result_path.with_name("config.yaml").write_text(
                "model: gpt2\nlearning_rate: .inf\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "companion config.*non-finite"):
                load_result_bundle(result_path)

            with self.assertRaisesRegex(ValueError, "result.*non-finite"):
                normalize_result_payload({"results": [{"method": "bad", "loss": float("inf")}]})

    def test_inline_config_supplements_companion_config_and_wins_on_conflicts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "run.json"
            result_path.write_text(
                json.dumps(
                    {
                        "results": [{"method": "uniform"}],
                        "config": {"model": "inline", "seed": 7},
                    }
                ),
                encoding="utf-8",
            )
            result_path.with_name("config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "model": "companion",
                        "dataset": "wikitext103",
                        "seed": 1,
                    }
                ),
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
            frame = manager._results_to_dataframe(
                [
                    {
                        "method": "top-level",
                        "loss": 1.5,
                        "perplexity": 4.5,
                        "accuracy": 0.0,
                    },
                    {
                        "method": "nested",
                        "metrics": {"loss": 2.5},
                    },
                ]
            )

            self.assertEqual(frame.loc[0, "metric_loss"], 1.5)
            self.assertEqual(frame.loc[0, "metric_perplexity"], 4.5)
            self.assertEqual(frame.loc[0, "metric_accuracy"], 0.0)
            self.assertEqual(frame.loc[1, "metric_loss"], 2.5)

    def test_result_dataframe_does_not_invent_missing_ratios_or_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = ResultManager(temp_dir)._results_to_dataframe([{"loss": 1.0}])

            self.assertTrue(frame["method"].isna().all())
            self.assertTrue(frame["importance"].isna().all())
            self.assertTrue(frame["allocation"].isna().all())
            self.assertTrue(frame["prune_ratio"].isna().all())
            self.assertTrue(frame["actual_ratio"].isna().all())

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
            primary_metric(
                {
                    "perplexity": 12.5,
                    "loss": 2.5,
                    "metrics": {"seconds": 0.4},
                }
            ),
            ("perplexity", 12.5),
        )
        self.assertEqual(
            primary_metric({"accuracy": 0.7, "metrics": {"accuracy": 0.7}}),
            ("accuracy", 0.7),
        )

    def test_result_metrics_rejects_conflicting_nested_metrics(self) -> None:
        with self.assertRaisesRegex(ValueError, "conflicting metric 'loss'"):
            result_metrics(
                {
                    "method": "example",
                    "loss": 9.0,
                    "metrics": {"loss": 2.5},
                }
            )
        self.assertEqual(
            result_metrics(
                {
                    "method": "example",
                    "loss": 9.0,
                    "perplexity": 12.5,
                    "metrics": {"loss": 9.0, "seconds": 0.4},
                }
            ),
            {"loss": 9.0, "perplexity": 12.5, "seconds": 0.4},
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
            (root / "wrapped_table1_run.json").write_text(
                json.dumps(
                    {
                        "results": [{"method": "wrapped", "target_ratio": 0.4, "perplexity": 12.0}],
                        "config": {"model": "pythia", "dataset": "alpaca"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary_ft_run.json").write_text(
                json.dumps(
                    {
                        "summary": {
                            "config": {
                                "model": "bert",
                                "dataset": "sst2",
                                "prune_ratio": 0.3,
                            },
                            "final": {"legacy": {"accuracy": 0.75}},
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "unrelated.json").write_text(
                json.dumps([{"distribution": "gamma", "shape": 2.0}]),
                encoding="utf-8",
            )

            collected = scan_results(root)
            self.assertEqual(
                {record["method"] for record in collected}, {"bare", "wrapped", "legacy"}
            )
            wrapped = next(record for record in collected if record["method"] == "wrapped")
            self.assertEqual(wrapped["prune_ratio"], 0.4)

            bundles = load_all_results(root)
            self.assertEqual(len(bundles), 3)
            entries = [entry for bundle in bundles for entry in extract_final_metrics(bundle)]
            self.assertEqual({entry["method"] for entry in entries}, {"bare", "wrapped", "legacy"})
            by_method = {entry["method"]: entry for entry in entries}
            self.assertEqual(by_method["bare"]["model"], "gpt2")
            self.assertEqual(by_method["wrapped"]["metric"], "perplexity")
            self.assertEqual(by_method["wrapped"]["prune_ratio"], 0.4)
            self.assertEqual(by_method["legacy"]["value"], 0.75)

    def test_aggregate_rejects_missing_or_invalid_numeric_evidence(self) -> None:
        valid_config = {"model": "gpt2", "dataset": "wiki", "prune_ratio": 0.3}
        cases = (
            ({"results": [{"method": "missing"}], "config": valid_config}, "no supported"),
            (
                {
                    "results": [{"method": "ratio", "loss": 1.0}],
                    "config": {**valid_config, "prune_ratio": 1.1},
                },
                "within \\[0, 1\\]",
            ),
            (
                {
                    "results": [{"method": "metric", "accuracy": "0.8"}],
                    "config": valid_config,
                },
                "finite number",
            ),
            (
                {
                    "results": [{"method": "metric", "accuracy": 1.1}],
                    "config": valid_config,
                },
                "within \\[0, 1\\]",
            ),
        )
        for payload, message in cases:
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(ResultScanError, message):
                    extract_final_metrics(payload)

    def test_aggregate_does_not_replace_missing_prune_ratio_with_a_default(self) -> None:
        payload = {
            "results": [{"method": "uniform", "loss": 1.0}],
            "config": {"model": "gpt2", "dataset": "wiki"},
        }

        with self.assertRaisesRegex(ResultScanError, "prune_ratio"):
            extract_final_metrics(payload)

    def test_lenient_scan_skips_malformed_files_but_keeps_valid_results(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "valid_table1_run.json").write_text(
                json.dumps(
                    {
                        "results": [{"method": "uniform", "loss": 1.0}],
                        "config": {"model": "gpt2", "dataset": "wiki", "prune_ratio": 0.3},
                    }
                ),
                encoding="utf-8",
            )
            (root / "broken_table1_run.json").write_text("{", encoding="utf-8")

            with self.assertRaisesRegex(ResultScanError, "broken_table1_run.json"):
                load_all_results(root)
            self.assertEqual(
                [bundle.source_path.name for bundle in load_all_results(root, strict=False)],
                ["valid_table1_run.json"],
            )


if __name__ == "__main__":
    unittest.main()
