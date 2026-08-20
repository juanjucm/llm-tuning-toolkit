import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune.guidellm import GuideLLMRunner, _TerminalScreen, extract_metrics
from auto_tune.tuner import AutoTuner


class GuideLLMAdapterTests(unittest.TestCase):
    def test_auto_tune_always_closes_the_display(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner._run_auto_tune = Mock(side_effect=RuntimeError("boom"))
        tuner._close_display = Mock()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            tuner.run_auto_tune()

        tuner._close_display.assert_called_once_with()

    def test_docker_runtime_arguments_are_passed_to_the_container(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner.config = {
            "port": 8000,
            "engine": {
                "image": "vllm/vllm-openai-rocm:latest",
                "docker": {
                    "devices": ["/dev/kfd:/dev/kfd:rwm", "/dev/dri:/dev/dri:rwm"],
                    "group_add": ["993"],
                    "security_opt": ["seccomp=unconfined"],
                    "environment": {"ROCR_VISIBLE_DEVICES": "0,1"},
                },
            },
        }
        tuner.hf_token = ""
        tuner.cache_dir = "/tmp/cache"
        tuner.logger = Mock()
        tuner.docker_client = Mock()

        tuner._launch_docker_engine(["--model", "example/model"])

        kwargs = tuner.docker_client.containers.run.call_args.kwargs
        self.assertEqual(kwargs["devices"], ["/dev/kfd:/dev/kfd:rwm", "/dev/dri:/dev/dri:rwm"])
        self.assertEqual(kwargs["group_add"], ["993"])
        self.assertEqual(kwargs["security_opt"], ["seccomp=unconfined"])
        self.assertEqual(kwargs["environment"]["ROCR_VISIBLE_DEVICES"], "0,1")
        self.assertEqual(kwargs["environment"]["HF_HUB_CACHE"], "/data/")
        self.assertNotIn("device_requests", kwargs)

    def test_existing_gpu_device_selection_is_preserved(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner.config = {
            "port": 8000,
            "engine": {"image": "vllm/vllm-openai:latest", "devices": ["0"]},
        }
        tuner.hf_token = ""
        tuner.cache_dir = "/tmp/cache"
        tuner.logger = Mock()
        tuner.docker_client = Mock()

        tuner._launch_docker_engine([])

        request = tuner.docker_client.containers.run.call_args.kwargs["device_requests"][0]
        self.assertEqual(request["DeviceIDs"], ["0"])
        self.assertEqual(request["Capabilities"], [["gpu"]])

    def test_load_profiles_map_to_their_guidellm_parameters(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner.config = {"scenario": {"load": {"kind": "throughput", "max_concurrency": 12}}}
        self.assertEqual(tuner._primary_profile(), {"kind": "throughput", "max_concurrency": 12})

        tuner.config["scenario"]["load"] = {"kind": "concurrent", "streams": 12}
        self.assertEqual(tuner._primary_profile(), {"kind": "concurrent", "streams": 12})

        tuner.config["scenario"]["load"] = {"kind": "constant", "rate": 8, "max_concurrency": 12}
        self.assertEqual(tuner._primary_profile(), {"kind": "constant", "rate": 8, "max_concurrency": 12})

        tuner.config["scenario"]["load"] = {"kind": "poisson", "rate": 8}
        self.assertEqual(tuner._primary_profile(), {"kind": "poisson", "rate": 8})

        tuner.config["scenario"]["load"] = {"kind": "replay", "time_scale": 0.5}
        self.assertEqual(tuner._primary_profile(), {"kind": "replay", "time_scale": 0.5})

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

            def fake_live_run(command, on_output):
                output_index = command.index("--output") + 1
                Path(json.loads(command[output_index])["path"]).write_text("{}")
                on_output("Benchmarks\nGenerating... 50%")
                return 0

            with (
                patch.object(GuideLLMRunner, "_run_with_live_output", side_effect=fake_live_run) as live_run,
                patch("auto_tune.guidellm.extract_metrics", return_value={"throughput": 5.0}),
            ):
                output = []
                metrics = GuideLLMRunner().run(
                    target="http://localhost:8000",
                    data=[{"kind": "json_file", "path": "tools.jsonl", "tool_choice": {"type": "function"}}],
                    profile={"kind": "constant", "rate": 5},
                    constraints=[{"kind": "max_requests", "count": 10}],
                    output_path=output_path,
                    on_output=output.append,
                )

            command = live_run.call_args.args[0]
            self.assertEqual(command[:2], ["guidellm", "run"])
            self.assertNotIn("--disable-console", command)
            self.assertNotIn("--disable-console-interactive", command)
            constraint_values = [command[index + 1] for index, value in enumerate(command) if value == "--constraint"]
            self.assertEqual(constraint_values, ['{"kind":"max_requests","count":10}'])
            self.assertEqual(metrics["throughput"], 5.0)
            self.assertTrue(output_path.exists())
            self.assertEqual(output, ["Benchmarks\nGenerating... 50%"])

    def test_terminal_screen_extracts_only_live_benchmark_progress(self):
        screen = _TerminalScreen(rows=12, columns=80)
        screen.feed(
            "static setup message\r\n"
            "╭─ Benchmarks ───────────────────────────────────────────────────────────────╮\r\n"
            "│ 10% concurrent  requests 10/100                                             │\r\n"
            "╰──────────────────────────────────────────────────────────────────────────────╯\r\n"
            "Generating... 10%"
        )

        progress = screen.benchmark_progress()
        self.assertIsNotNone(progress)
        self.assertIn("Benchmarks", progress)
        self.assertIn("Generating... 10%", progress)
        self.assertNotIn("static setup message", progress)

        screen.feed("\r\x1b[2KGenerating... 50%")
        self.assertIn("Generating... 50%", screen.benchmark_progress())

    def test_runner_inherits_terminal_output_without_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"

            def fake_run(command, **_):
                output_index = command.index("--output") + 1
                Path(json.loads(command[output_index])["path"]).write_text("{}")

            with (
                patch("auto_tune.guidellm.subprocess.run", side_effect=fake_run) as run,
                patch("auto_tune.guidellm.subprocess.Popen") as popen,
                patch("auto_tune.guidellm.extract_metrics", return_value={"throughput": 5.0}),
            ):
                metrics = GuideLLMRunner().run(
                    target="http://localhost:8000",
                    data=[{"kind": "synthetic_text", "prompt_tokens": 10, "output_tokens": 2}],
                    profile={"kind": "throughput", "max_concurrency": 4},
                    constraints=[{"kind": "max_requests", "count": 10}],
                    output_path=output_path,
                )

            command = run.call_args.args[0]
            self.assertNotIn("--disable-console-interactive", command)
            self.assertEqual(run.call_args.kwargs, {"check": True})
            popen.assert_not_called()
            self.assertEqual(metrics["throughput"], 5.0)


if __name__ == "__main__":
    unittest.main()
