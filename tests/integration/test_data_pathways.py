import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

import pytest
from guidellm.benchmark import GenerativeBenchmarksReport

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from auto_tune.guidellm import GuideLLMRunner

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_MODEL = "Qwen/Qwen3-0.6B"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.integration
@unittest.skipIf(sys.platform == "darwin", "guidellm 0.7.x workers segfault on macOS (fork start method)")
class DataPathwayTests(unittest.TestCase):
    """Run real guidellm against its mock server through GuideLLMRunner."""

    @classmethod
    def setUpClass(cls):
        cls.port = _free_port()
        cls.target = f"http://127.0.0.1:{cls.port}"
        cls.guidellm_bin = str(Path(sys.executable).parent / "guidellm")
        cls.server_log = tempfile.NamedTemporaryFile(suffix=".log", delete=False)
        cls.server = subprocess.Popen(
            [cls.guidellm_bin, "mock-server", "--host", "127.0.0.1",
             "--port", str(cls.port), "--model", MOCK_MODEL],
            stdout=cls.server_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if cls.server.poll() is not None:
                raise RuntimeError(f"mock server exited early; see {cls.server_log.name}")
            try:
                with urllib.request.urlopen(f"{cls.target}/health", timeout=2) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("mock server not ready in 120 s")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        try:
            cls.server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            cls.server.kill()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_path = Path(self.tmp.name) / "report.json"
        self.runner = GuideLLMRunner(command=self.guidellm_bin)

    def tearDown(self):
        self.tmp.cleanup()

    def _report(self) -> GenerativeBenchmarksReport:
        return GenerativeBenchmarksReport.load_file(str(self.output_path))

    def test_sharegpt_json_file_pathway(self):
        metrics = self.runner.run(
            target=self.target,
            data=[{
                "kind": "json_file",
                "path": str(FIXTURES / "sharegpt_sample.json"),
                "load_kwargs": {"split": "train"},  # required on guidellm 0.7.3
            }],
            profile={"kind": "throughput", "max_concurrency": 8},
            constraints=[{"kind": "max_requests", "count": 16}],
            output_path=self.output_path,
            options={
                "sample_size": 0,
                "arguments": {
                    "data-column-mapper": {
                        "kind": "generative_column_mapper",
                        "column_mappings": {"text_column": "conversations"},
                    },
                    "data-preprocessor": {"kind": "tool_calling_message_extractor"},
                },
            },
            show_console=False,
        )
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertGreater(metrics["throughput"], 0)
        self.assertIn("ttft_p99_ms", metrics)

        benchmark = self._report().benchmarks[0]
        prompt_mean = benchmark.metrics.prompt_token_count.successful.mean
        self.assertGreater(prompt_mean, 20)
        self.assertLess(prompt_mean, 2000)

    def test_huggingface_dataset_with_column_mapper(self):
        metrics = self.runner.run(
            target=self.target,
            data=[{
                "kind": "huggingface",
                "source": "garage-bAInd/Open-Platypus",
                "load_kwargs": {"split": "train"},
            }],
            profile={"kind": "throughput", "max_concurrency": 8},
            constraints=[{"kind": "max_requests", "count": 16}],
            output_path=self.output_path,
            options={
                "sample_size": 0,
                "arguments": {
                    "data-column-mapper": {
                        "kind": "generative_column_mapper",
                        "column_mappings": {"text_column": "instruction"},
                    },
                    "data-loader": {"kind": "pytorch", "samples": 100},
                },
            },
            show_console=False,
        )
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertGreater(metrics["throughput"], 0)

    def test_trace_replay_honors_timestamps(self):
        span = 10.9
        rows = [
            {"timestamp": burst + i * 0.1, "input_length": 128, "output_length": 32}
            for burst in (0.0, 5.0, 10.0)
            for i in range(10)
        ]
        trace_path = Path(self.tmp.name) / "trace.jsonl"
        with trace_path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        metrics = self.runner.run(
            target=self.target,
            data=[f"kind=trace_synthetic,path={trace_path}"],
            profile={"kind": "replay", "time_scale": 1.0},
            constraints=[{"kind": "max_duration", "seconds": 300}],
            output_path=self.output_path,
            options={
                "sample_size": 0,
                "arguments": {
                    "tokenizer": {"kind": "huggingface_auto", "model": MOCK_MODEL},
                },
            },
            show_console=False,
        )
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertEqual(int(metrics["successful_requests"]), len(rows))

        benchmark = self._report().benchmarks[0]
        self.assertGreaterEqual(benchmark.duration, span * 0.9)


if __name__ == "__main__":
    unittest.main()
