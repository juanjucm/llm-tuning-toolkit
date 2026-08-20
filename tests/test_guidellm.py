import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune.guidellm import GuideLLMRunner, extract_metrics
from auto_tune.tuner import AutoTuner


class GuideLLMAdapterTests(unittest.TestCase):
    def test_load_profiles_map_to_their_guidellm_parameters(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner.config = {"scenario": {"load": {"kind": "throughput", "max_concurrency": 12}}}
        self.assertEqual(tuner._primary_profile(), {"kind": "throughput", "max_concurrency": 12})

        tuner.config["scenario"]["load"] = {"kind": "concurrent", "streams": 12}
        self.assertEqual(tuner._primary_profile(), {"kind": "concurrent", "streams": 12})

    def test_extract_metrics_uses_highest_throughput_strategy(self):
        def summary(mean, p99):
            return SimpleNamespace(
                successful=SimpleNamespace(
                    mean=mean,
                    percentiles=SimpleNamespace(
                        p50=mean,
                        p75=mean,
                        p90=mean,
                        p95=mean,
                        p99=p99,
                    ),
                )
            )

        metrics_model = SimpleNamespace(
            requests_per_second=summary(18.0, 20.0),
            request_totals=SimpleNamespace(successful=20, errored=0, incomplete=0, total=20),
            output_tokens_per_second=summary(42.0, 43.0),
            tokens_per_second=summary(84.0, 85.0),
            time_to_first_token_ms=summary(10.0, 20.0),
            request_latency=summary(0.03, 0.05),
            inter_token_latency_ms=summary(1.5, 4.0),
        )
        report = SimpleNamespace(benchmarks=[SimpleNamespace(metrics=metrics_model)])

        with patch("auto_tune.guidellm.GenerativeBenchmarksReport.model_validate", return_value=report):
            metrics = extract_metrics({"metadata": {"version": 2}})

        self.assertEqual(metrics["throughput"], 18.0)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["ttft_p99_ms"], 20.0)
        self.assertEqual(metrics["e2e_p99_ms"], 50.0)
        self.assertNotIn("ttft_p60_ms", metrics)

    def test_runner_serializes_nested_data_and_writes_to_requested_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"

            def fake_run(command, **_):
                output_index = command.index("--output") + 1
                Path(json.loads(command[output_index])["path"]).write_text("{}")

            with (
                patch("auto_tune.guidellm.subprocess.run", side_effect=fake_run) as run,
                patch("auto_tune.guidellm.extract_metrics", return_value={"throughput": 5.0}),
            ):
                metrics = GuideLLMRunner().run(
                    target="http://localhost:8000",
                    data=[{"kind": "json_file", "path": "tools.jsonl", "tool_choice": {"type": "function"}}],
                    profile={"kind": "constant", "rate": 5},
                    duration_seconds=10,
                    output_path=output_path,
                )

            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["guidellm", "run"])
            self.assertNotIn("--disable-console", command)
            self.assertNotIn("capture_output", run.call_args.kwargs)
            self.assertEqual(metrics["throughput"], 5.0)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
