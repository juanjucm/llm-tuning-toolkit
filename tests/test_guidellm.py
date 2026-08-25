import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml
from guidellm.schemas.statistics import Percentiles
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from auto_tune.display import AutoTuneDisplay
from auto_tune.display_format import format_workload
from auto_tune.guidellm import GuideLLMRunner, extract_all_metrics, extract_metrics
from auto_tune.terminal import TerminalScreen
from auto_tune.tuner import AutoTuner, evaluate_slos


def load_scenario(**scenario) -> dict:
    """Run a minimal config through _load_config so tests see production shapes.

    Passing a key as None drops it, which is how the absent-key cases are expressed.
    """
    merged = {
        "name": "s",
        "data": [{"kind": "synthetic_text", "prompt_tokens": 8}],
        "slos": {"min_success_rate": 0.99},
        **scenario,
    }
    config = {
        "model": "m",
        "port": 8000,
        "instance_info": {"gpu_type": "t"},
        "engine": {"name": "vllm", "image": "img", "base_args": []},
        "scenario": {key: value for key, value in merged.items() if value is not None},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        config_path = handle.name

    tuner = AutoTuner.__new__(AutoTuner)
    tuner.config_path = config_path
    return tuner._load_config()


def latency_summary(mean: float, p99: float) -> SimpleNamespace:
    """A GuideLLM latency summary carrying the exact percentile set it publishes."""
    values = dict.fromkeys(Percentiles.model_fields, float(mean))
    values["p99"] = float(p99)
    return SimpleNamespace(successful=SimpleNamespace(mean=mean, percentiles=Percentiles(**values)))


def benchmark(throughput: float, ttft_p99: float) -> SimpleNamespace:
    """A GenerativeBenchmark stand-in describing one measured load point."""
    return SimpleNamespace(
        metrics=SimpleNamespace(
            requests_per_second=latency_summary(throughput, throughput),
            request_totals=SimpleNamespace(successful=20, errored=0, incomplete=0, total=20),
            output_tokens_per_second=latency_summary(42.0, 43.0),
            tokens_per_second=latency_summary(84.0, 85.0),
            time_to_first_token_ms=latency_summary(10.0, ttft_p99),
            request_latency=latency_summary(0.03, 0.05),
            inter_token_latency_ms=latency_summary(1.5, 4.0),
        )
    )


class GuideLLMAdapterTests(unittest.TestCase):
    def test_display_orders_run_state_and_renders_best_metrics(self):
        display = AutoTuneDisplay()
        display._live = Mock()
        display.start(
            total_configs=4,
            model="org/model",
            scenario_name="balanced",
            scenario={
                "load": {"kind": "throughput", "max_concurrency": 128},
                "constraints": [{"kind": "max_requests", "count": 1000}],
                "data": [{"kind": "synthetic_text", "prompt_tokens": 1024, "output_tokens": 128}],
                "slos": {"min_success_rate": 0.99, "max_ttft_p99_ms": 250},
            },
        )
        display.progress.update(display._task_id, completed=1)
        parameters = {"value_args": {"max-num-seqs": 512}, "action_args": {"prefix-caching": True}}
        display.begin_config(2, 4, parameters)
        display.set_status("Running GuideLLM throughput benchmark")
        display.set_best(
            {
                "throughput": 3.79,
                "output_tokens_per_second": 7986.5,
                "success_rate": 0.99,
                "ttft_avg_ms": 42.0,
                "ttft_p99_ms": 84.0,
                "itl_avg_ms": 8.0,
                "itl_p99_ms": 14.0,
                "e2e_avg_ms": 2200.0,
                "e2e_p99_ms": 3100.0,
            },
            parameters,
        )

        output = io.StringIO()
        console = Console(file=output, width=180)
        console.print(display._render())
        rendered = output.getvalue()

        for expected in (
            "org/model",
            "balanced",
            "Throughput · max concurrency 128",
            "Max requests: 1,000",
            "Synthetic text · input 1024 tok · output 128 tok",
            "Success rate  ≥ 99.0%",
            "TTFT p99  ≤ 250 ms",
            "Best run",
            "3.79 req/s",
            "42.0 / 84.0 ms",
            "Live benchmark",
        ):
            self.assertIn(expected, rendered)
        self.assertLess(rendered.index("Configurations"), rendered.index("Current"))
        self.assertLess(rendered.index("Current"), rendered.index("Status"))
        self.assertLess(rendered.index("Status"), rendered.index("Best run"))

    def test_display_formats_every_supported_workload_profile(self):
        # Feed each load through _load_config so the rendered shapes are the ones
        # production hands to the display, not hand-written approximations.
        cases = [
            ({"kind": "throughput", "max_concurrency": 64}, "Throughput · max concurrency 64"),
            ({"kind": "concurrent", "streams": 32}, "Concurrent · 32 streams"),
            ({"kind": "concurrent", "streams": [1, 4, 16]}, "Concurrent · 1, 4, 16 streams"),
            (
                {"kind": "constant", "rate": 12, "max_concurrency": 100},
                "Constant rate · 12 req/s · max concurrency 100",
            ),
            ({"kind": "poisson", "rate": 8}, "Poisson arrivals · 8 req/s"),
            ({"kind": "replay", "time_scale": 0.5}, "Trace replay · 0.5× speed"),
            ({"kind": "constant", "rate": 4}, "Constant rate · 4 req/s"),
        ]
        for load, expected in cases:
            with self.subTest(load=load):
                self.assertEqual(format_workload(load_scenario(load=load)["scenario"]["load"]), expected)

    def test_auto_tune_always_closes_the_display(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner._run_auto_tune = Mock(side_effect=RuntimeError("boom"))
        tuner._close_display = Mock()

        with self.assertRaisesRegex(RuntimeError, "boom"):
            tuner.run_auto_tune()

        tuner._close_display.assert_called_once_with()

    def test_rate_finding_uses_every_configured_attempt(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner.config = {
            "scenario": {
                "max_rate_finding_attempts": 3,
                "rate_decrease_factor": 0.25,
                "slos": {"min_success_rate": 0.99},
            }
        }
        tuner.logger = Mock()
        tuner.tracker = Mock()
        tuner._run_rate_benchmark = Mock(return_value={"throughput": 10.0, "success_rate": 0.5})

        with patch("auto_tune.tuner.time.sleep"):
            metrics, checks = tuner._find_optimal_rate(10.0, "run", "results")

        self.assertIsNone(metrics)
        self.assertIsNone(checks)
        self.assertEqual(tuner._run_rate_benchmark.call_count, 3)

    def test_container_cleanup_forces_removal_when_stop_fails(self):
        tuner = AutoTuner.__new__(AutoTuner)
        tuner.logger = Mock()
        tuner.display = None
        container = Mock()
        container.stop.side_effect = RuntimeError("stop failed")

        tuner._cleanup_container(container)

        container.remove.assert_called_once_with(force=True)

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

    def test_load_config_accepts_only_the_kinds_auto_tune_tunes(self):
        # GuideLLM's concurrent profile takes a list of stream counts.
        self.assertEqual(load_scenario(load={"kind": "concurrent", "streams": 12})["scenario"]["load"]["streams"], [12])
        self.assertEqual(load_scenario(load={"kind": "replay"})["scenario"]["load"]["time_scale"], 1.0)

        with self.assertRaises(ValueError) as unknown:
            load_scenario(load={"kind": "nonsense"})
        self.assertIn("is not a GuideLLM profile", str(unknown.exception))
        # Declined kinds must not be advertised as supported.
        self.assertIn(
            "Supported: concurrent, constant, poisson, replay, throughput",
            str(unknown.exception),
        )

        # Each declined kind names what to do instead. `async` carries a rate, so its
        # rejection is about the alias rather than a missing field.
        for load, guidance in (
            ({"kind": "async", "rate": 5}, "use 'constant' or 'poisson' with an explicit rate"),
            ({"kind": "synchronous"}, "measures a latency baseline"),
            ({"kind": "sweep"}, "searches rates itself"),
        ):
            with self.subTest(kind=load["kind"]), self.assertRaises(ValueError) as declined:
                load_scenario(load=load)
            self.assertIn("is not supported", str(declined.exception))
            self.assertIn(guidance, str(declined.exception))

        with self.assertRaises(ValueError) as rateless:
            load_scenario(load={"kind": "constant"})
        self.assertIn("scenario.load.rate is required", str(rateless.exception))

    def test_load_config_rejects_slos_no_metric_can_satisfy(self):
        with self.assertRaises(ValueError) as unsupported:
            load_scenario(slos={"max_ttft_p60_ms": 500})
        self.assertIn("unsupported metric 'ttft_p60_ms'", str(unsupported.exception))

        # A percentile GuideLLM does report must be usable.
        self.assertEqual(load_scenario(slos={"max_ttft_p75_ms": 500})["scenario"]["slos"], {"max_ttft_p75_ms": 500})

        # A misspelled `slos:` key leaves the mapping empty, which used to accept
        # every configuration because all([]) is True.
        with self.assertRaises(ValueError) as empty:
            load_scenario(slos={})
        self.assertIn("must be a non-empty mapping", str(empty.exception))

        with self.assertRaises(ValueError) as unprefixed:
            load_scenario(slos={"ttft_p99_ms": 500})
        self.assertIn("must start with min_ or max_", str(unprefixed.exception))

        # The legacy name still migrates, but not alongside its replacement.
        migrated = load_scenario(slos=None, goodput_criteria={"max_ttft_p99_ms": 500})
        self.assertEqual(migrated["scenario"]["slos"], {"max_ttft_p99_ms": 500})
        with self.assertRaises(ValueError) as both:
            load_scenario(slos={"max_ttft_p99_ms": 500}, goodput_criteria={"max_ttft_p99_ms": 500})
        self.assertIn("remove the legacy scenario.goodput_criteria key", str(both.exception))

    def test_extract_metrics_uses_highest_throughput_strategy(self):
        report = SimpleNamespace(benchmarks=[benchmark(4.0, 900.0), benchmark(18.0, 20.0)])

        with patch("auto_tune.guidellm.GenerativeBenchmarksReport.model_validate", return_value=report):
            metrics = extract_metrics({"metadata": {"version": 2}})

        self.assertEqual(metrics["throughput"], 18.0)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(metrics["ttft_p99_ms"], 20.0)
        self.assertEqual(metrics["e2e_p99_ms"], 50.0)
        # The percentile set is GuideLLM's, so p75 is reachable and p60 does not exist.
        self.assertIn("ttft_p75_ms", metrics)
        self.assertNotIn("ttft_p60_ms", metrics)

    def test_lower_load_point_wins_when_the_saturated_one_misses_slos(self):
        report = SimpleNamespace(benchmarks=[benchmark(40.0, 900.0), benchmark(10.0, 100.0)])

        with patch("auto_tune.guidellm.GenerativeBenchmarksReport.model_validate", return_value=report):
            candidates = extract_all_metrics({"metadata": {"version": 2}})

        self.assertEqual([point["throughput"] for point in candidates], [10.0, 40.0])

        tuner = AutoTuner.__new__(AutoTuner)
        tuner.config = {"scenario": {"slos": {"max_ttft_p99_ms": 500}}}
        metrics, checks, meets = tuner._select_load_point(candidates)
        self.assertTrue(meets)
        self.assertEqual(metrics["throughput"], 10.0)
        self.assertEqual(checks, [{"max_ttft_p99_ms": 500, "ttft_p99_ms": 100.0, "meets": True}])

        # With no qualifying point, the most saturated one is still reported.
        tuner.config["scenario"]["slos"] = {"max_ttft_p99_ms": 50}
        metrics, _, meets = tuner._select_load_point(candidates)
        self.assertFalse(meets)
        self.assertEqual(metrics["throughput"], 40.0)

    def test_evaluate_slos_compares_both_directions(self):
        metrics = {"throughput": 12.0, "success_rate": 0.5, "ttft_p99_ms": 400.0}
        checks, meets = evaluate_slos({"min_success_rate": 0.99, "max_ttft_p99_ms": 500}, metrics)
        self.assertFalse(meets)
        self.assertEqual([check["meets"] for check in checks], [False, True])

        with self.assertRaises(ValueError):
            evaluate_slos({"max_missing_metric": 1}, metrics)

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
        screen = TerminalScreen(rows=12, columns=80)
        screen.feed(
            "static setup message\r\n"
            "╭─ Benchmarks ───────────────────────────────────────────────────────────────╮\r\n"
            "│ 10% concurrent  requests 10/100                                             │\r\n"
            "╰──────────────────────────────────────────────────────────────────────────────╯\r\n"
            "Generating... 10%"
        )

        progress = screen.extract_block("Benchmarks", "Generating...")
        self.assertIsNotNone(progress)
        self.assertIn("Benchmarks", progress)
        self.assertIn("Generating... 10%", progress)
        self.assertNotIn("static setup message", progress)

        screen.feed("\r\x1b[2KGenerating... 50%")
        self.assertIn("Generating... 50%", screen.extract_block("Benchmarks", "Generating..."))

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

    def test_runner_suppresses_all_console_output_when_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "result.json"

            def fake_run(command, **_):
                output_index = command.index("--output") + 1
                Path(json.loads(command[output_index])["path"]).write_text("{}")

            with (
                patch("auto_tune.guidellm.subprocess.run", side_effect=fake_run) as run,
                patch("auto_tune.guidellm.extract_metrics", return_value={"throughput": 5.0}),
            ):
                GuideLLMRunner().run(
                    target="http://localhost:8000",
                    data=[{"kind": "synthetic_text", "prompt_tokens": 10, "output_tokens": 2}],
                    profile={"kind": "throughput", "max_concurrency": 4},
                    constraints=[{"kind": "max_requests", "count": 10}],
                    output_path=output_path,
                    show_console=False,
                )

            command = run.call_args.args[0]
            self.assertIn("--disable-console", command)
            self.assertIn("--disable-console-interactive", command)
            self.assertEqual(
                run.call_args.kwargs,
                {"check": True, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL},
            )


if __name__ == "__main__":
    unittest.main()
