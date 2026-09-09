import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import yaml
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune.benchmarking import BenchmarkSuite
from auto_tune.benchmarking import cli as benchmark_cli
from auto_tune.benchmarking.config import load_suite
from auto_tune.guidellm import GuideLLMError


def suite_data() -> dict:
    workload = {
        "data": [{"kind": "synthetic_text", "prompt_tokens": 1024, "output_tokens": 128}],
        "profile": {"kind": "concurrent", "streams": [1, 64]},
        "constraints": [{"kind": "max_requests", "count": 100}],
    }
    return {
        "name": "production",
        "description": "Serving characterization",
        "backend": {"kind": "openai_http", "request_format": "/v1/chat/completions"},
        "guidellm_options": {"sample_size": 0, "arguments": {"seed": {"kind": "static", "value": 42}}},
        "benchmarks": [
            {
                "name": "high-concurrency",
                "description": "Closed-loop concurrency",
                "tags": ["core", "concurrency"],
                **workload,
                "slos": {"min_success_rate": 0.99, "max_ttft_p99_ms": 500},
            },
            {
                "name": "long-context",
                "description": "Long prefill",
                "tags": ["core", "long-context"],
                "selection": {"min_context_window": 32768},
                **workload,
            },
            {
                "name": "vision-chat",
                "description": "Vision requests",
                "tags": ["optional", "vision"],
                "selection": {"capabilities": ["vision"]},
                **workload,
            },
        ],
    }


def measured_point(streams: int, throughput: float, ttft_p99_ms: float) -> dict:
    return {
        "load": {"kind": "concurrent", "streams": streams, "max_concurrency": streams},
        "metrics": {
            "throughput": throughput,
            "success_rate": 1.0,
            "ttft_p99_ms": ttft_p99_ms,
            "e2e_p99_ms": 1000.0,
        },
    }


class BenchmarkSuiteTests(unittest.TestCase):
    def create_runner(
        self,
        directory: str,
        *,
        config: dict | None = None,
        tokenizer_model: str | None = None,
        context_window: int | None = 8192,
        capabilities: tuple[str, ...] = (),
        benchmark_names: tuple[str, ...] = (),
        include_tags: tuple[str, ...] = (),
        exclude_tags: tuple[str, ...] = (),
        hf_token: str | None = None,
    ) -> BenchmarkSuite:
        config_path = Path(directory) / "suite.yaml"
        config_path.write_text(yaml.safe_dump(config or suite_data(), sort_keys=False))
        environment = {"HF_TOKEN": hf_token} if hf_token is not None else {}
        with patch.dict(os.environ, environment, clear=True):
            return BenchmarkSuite(
                str(config_path),
                recipe="vllm-l40s-fp8",
                target="http://serving.internal:8000",
                model="org/model",
                tokenizer_model=tokenizer_model,
                result_dir=directory,
                context_window=context_window,
                capabilities=capabilities,
                benchmark_names=benchmark_names,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                show_console=False,
            )

    def test_caller_managed_recipe_runs_without_deployment_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.create_runner(directory, tokenizer_model="org/tokenizer")
            runner.guidellm = Mock()
            runner.guidellm.run_points.return_value = [
                measured_point(1, 4.0, 100.0),
                measured_point(64, 40.0, 900.0),
            ]

            summary = runner.run()

            self.assertEqual(summary["recipe"], "vllm-l40s-fp8")
            self.assertEqual(summary["target"], "http://serving.internal:8000")
            self.assertEqual(
                [item["status"] for item in summary["benchmarks"]], ["completed", "skipped", "skipped"]
            )
            high_concurrency = summary["benchmarks"][0]
            self.assertEqual([point["load"]["streams"] for point in high_concurrency["load_points"]], [1, 64])
            self.assertEqual([point["slo_met"] for point in high_concurrency["load_points"]], [True, False])
            call = runner.guidellm.run_points.call_args.kwargs
            self.assertEqual(
                call["backend"],
                {
                    "kind": "openai_http",
                    "request_format": "/v1/chat/completions",
                    "target": "http://serving.internal:8000",
                    "model": "org/model",
                },
            )
            self.assertEqual(
                call["options"]["arguments"]["tokenizer"],
                {"kind": "huggingface_auto", "model": "org/tokenizer"},
            )
            persisted = json.loads(Path(summary["report_path"]).read_text())
            self.assertEqual(persisted["recipe"], "vllm-l40s-fp8")
            self.assertNotIn("engine", persisted)

    def test_exported_hf_token_authenticates_the_endpoint_without_leaking_into_results(self):
        with tempfile.TemporaryDirectory() as directory:
            token = "hf_private_endpoint_token"
            runner = self.create_runner(
                directory,
                benchmark_names=("high-concurrency",),
                hf_token=token,
            )
            runner.guidellm = Mock()
            runner.guidellm.run_points.return_value = [measured_point(1, 4.0, 100.0)]

            summary = runner.run()

            backend = runner.guidellm.run_points.call_args.kwargs["backend"]
            self.assertEqual(backend["api_key"], token)
            self.assertEqual(
                summary["authentication"],
                {"kind": "huggingface_token", "enabled": True},
            )
            self.assertNotIn(token, Path(summary["report_path"]).read_text())

    def test_model_traits_enable_only_applicable_optional_workloads(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.create_runner(
                directory,
                context_window=32768,
                capabilities=("vision",),
            )
            runner.guidellm = Mock()
            runner.guidellm.run_points.return_value = [measured_point(1, 4.0, 100.0)]

            summary = runner.run()

            self.assertEqual([item["status"] for item in summary["benchmarks"]], ["completed"] * 3)
            self.assertEqual(runner.guidellm.run_points.call_count, 3)

    def test_caller_filters_are_explicit_and_unknown_names_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.create_runner(directory, benchmark_names=("high-concurrency",))
            runner.guidellm = Mock()
            runner.guidellm.run_points.return_value = [measured_point(1, 4.0, 100.0)]
            summary = runner.run()
            self.assertEqual(
                [item["status"] for item in summary["benchmarks"]], ["completed", "skipped", "skipped"]
            )
            self.assertEqual(summary["benchmarks"][1]["reason"], "not selected by caller")

            unknown = self.create_runner(directory, benchmark_names=("missing",))
            with self.assertRaisesRegex(ValueError, "Unknown benchmark names: missing"):
                unknown.run()

    def test_failed_scenario_is_reported_without_skipping_later_work(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self.create_runner(directory, context_window=32768)
            runner.guidellm = Mock()
            runner.guidellm.run_points.side_effect = [
                GuideLLMError("backend rejected requests"),
                [measured_point(1, 1.0, 300.0)],
            ]

            summary = runner.run()

            self.assertEqual(
                [item["status"] for item in summary["benchmarks"]], ["failed", "completed", "skipped"]
            )
            self.assertEqual(summary["benchmarks"][0]["error"], "backend rejected requests")

    def test_suite_cannot_override_runtime_backend_credentials_or_routing(self):
        for field, value in (
            ("target", "http://wrong.example"),
            ("model", "wrong/model"),
            ("api_key", "hardcoded-secret"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config = suite_data()
                config["backend"][field] = value
                config_path = Path(directory) / "suite.yaml"
                config_path.write_text(yaml.safe_dump(config, sort_keys=False))

                with self.assertRaisesRegex(ValueError, "runtime-supplied"):
                    load_suite(config_path)

    def test_production_example_is_valid_and_model_aware(self):
        example = Path(__file__).parents[1] / "examples" / "guidellm-recipe-benchmark.yaml"
        suite = load_suite(example)

        self.assertEqual(suite.name, "production-serving")
        self.assertEqual(len(suite.benchmarks), 7)
        self.assertEqual(
            suite.model_dump()["benchmarks"][3]["selection"]["min_context_window"],
            32768,
        )
        self.assertEqual(
            suite.model_dump()["benchmarks"][6]["selection"]["capabilities"],
            frozenset({"vision"}),
        )


class BenchmarkCliTests(unittest.TestCase):
    def test_cli_forwards_recipe_endpoint_and_model_selection_arguments(self):
        runner = CliRunner()
        suite = Mock()
        suite.run.return_value = {"benchmarks": [{"status": "completed"}]}
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "suite.yaml"
            config.write_text("name: placeholder\n")
            with patch("auto_tune.benchmarking.cli.BenchmarkSuite", return_value=suite) as benchmark_suite:
                result = runner.invoke(
                    benchmark_cli.app,
                    [
                        "--config",
                        str(config),
                        "--recipe",
                        "recipe-42",
                        "--target",
                        "https://serving.example/v1",
                        "--model",
                        "org/model",
                        "--tokenizer-model",
                        "org/tokenizer",
                        "--context-window",
                        "32768",
                        "--capability",
                        "vision",
                        "--benchmark",
                        "vision-chat",
                        "--include-tag",
                        "optional",
                        "--exclude-tag",
                        "expensive",
                        "--result-dir",
                        directory,
                        "--no-console",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        benchmark_suite.assert_called_once_with(
            config_path=str(config.resolve()),
            recipe="recipe-42",
            target="https://serving.example/v1",
            model="org/model",
            tokenizer_model="org/tokenizer",
            result_dir=directory,
            context_window=32768,
            capabilities=["vision"],
            benchmark_names=["vision-chat"],
            include_tags=["optional"],
            exclude_tags=["expensive"],
            guidellm_command="guidellm",
            show_console=False,
        )
        suite.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
