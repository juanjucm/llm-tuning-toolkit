import importlib.util
import logging
import subprocess
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


if importlib.util.find_spec("docker") is None:
    docker_module = types.ModuleType("docker")
    docker_module.from_env = lambda: None
    docker_module.types = types.SimpleNamespace(DeviceRequest=lambda *args, **kwargs: (args, kwargs))
    sys.modules["docker"] = docker_module
    sys.modules["docker.models"] = types.ModuleType("docker.models")
    containers_module = types.ModuleType("docker.models.containers")
    containers_module.Container = object
    sys.modules["docker.models.containers"] = containers_module

if importlib.util.find_spec("requests") is None:
    requests_module = types.ModuleType("requests")
    requests_module.RequestException = Exception
    requests_module.get = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_module

import benchmarker.executors as executor_module
from benchmarker.engine import normalize_devices
from benchmarker.engine import spec_from_engine_config
from benchmarker.executors import BenchmarkRequest
from benchmarker.executors import BenchmarkRun
from benchmarker.executors import InferenceBenchmarkerExecutor
from benchmarker.executors import create_benchmark_executor
from benchmarker.executors import serialize_metadata
from benchmarker.results import extract_inference_benchmarker_metrics


def test_inference_benchmarker_command_contract():
    calls = {}
    original_run = executor_module.subprocess.run

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    executor_module.subprocess.run = fake_run
    try:
        executor = InferenceBenchmarkerExecutor(logger=logging.getLogger("test"))
        executor.run(BenchmarkRun(args=["--url", "http://localhost:80"], extra_meta={"engine": "vllm"}))
    finally:
        executor_module.subprocess.run = original_run

    assert calls["cmd"] == [
        "inference-benchmarker",
        "--url",
        "http://localhost:80",
        "--extra-meta",
        "engine=vllm",
        "--no-console",
    ]
    assert calls["kwargs"]["capture_output"] is True
    assert calls["kwargs"]["text"] is True
    assert calls["kwargs"]["check"] is True


def test_inference_benchmarker_structured_request_contract():
    executor = InferenceBenchmarkerExecutor(logger=logging.getLogger("test"))

    run = executor.build_request_run(
        BenchmarkRequest(
            kind="rate",
            url="http://localhost:80",
            max_vus=128,
            duration="30s",
            tokenizer_name="test/model",
            output_path="rate.json",
            run_id="abc1",
            prompt_options="num_tokens=128",
            decode_options="num_tokens=32",
            dataset_file="dataset.json",
            rate=12.5,
            extra_meta={"engine": "vllm"},
        )
    )

    assert run.extra_meta == {"engine": "vllm"}
    assert run.args == [
        "--url",
        "http://localhost:80",
        "--benchmark-kind",
        "rate",
        "--max-vus",
        128,
        "--duration",
        "30s",
        "--tokenizer-name",
        "test/model",
        "--output-path",
        "rate.json",
        "--run-id",
        "abc1",
        "--rates",
        12.5,
        "--prompt-options",
        "num_tokens=128",
        "--decode-options",
        "num_tokens=32",
        "--dataset-file",
        "dataset.json",
    ]


def test_vllm_bench_command_contract():
    executor = create_benchmark_executor("vllm-bench", command="python -m fake_bench")

    cmd = executor.build_command(BenchmarkRun(args=["--host", "127.0.0.1"], extra_meta={"ignored": "yes"}))

    assert cmd == ["python", "-m", "fake_bench", "--host", "127.0.0.1"]


def test_metadata_serialization_handles_structured_values():
    metadata = serialize_metadata(
        {
            "engine": "vllm",
            "instance": {"gpu": "L40S", "count": 4},
            "description": "balanced, throughput=SLO",
            "empty": None,
        }
    )

    assert "engine=vllm" in metadata
    assert "instance=" in metadata
    assert "description=balanced%2C%20throughput%3DSLO" in metadata
    assert "empty=" not in metadata


def test_engine_config_normalizes_devices_and_env():
    assert normalize_devices(["all"]) == "all"
    assert normalize_devices("0,1") == ["0", "1"]

    spec = spec_from_engine_config(
        {
            "name": "vLLM",
            "image": "vllm/vllm-openai:test",
            "devices": ["all"],
            "envs": ["A=B", "IGNORED"],
            "args": ["--model", "test/model"],
        },
        8000,
        extra_env={"HF_TOKEN": "token"},
    )

    assert spec.name == "vLLM"
    assert spec.image == "vllm/vllm-openai:test"
    assert spec.command == ["--model", "test/model"]
    assert spec.devices == "all"
    assert spec.environment == {"A": "B", "HF_TOKEN": "token"}


def test_inference_benchmarker_metrics_contract():
    metrics = extract_inference_benchmarker_metrics(
        {
            "results": [
                {
                    "request_rate": 12.5,
                    "total_requests": 10,
                    "successful_requests": 9,
                    "failed_requests": 1,
                    "time_to_first_token_ms": {"p99": 101, "avg": 50},
                    "e2e_latency_ms": {"p99": 202, "avg": 100},
                    "inter_token_latency_ms": {"p99": 11, "avg": 5},
                }
            ]
        }
    )

    assert metrics["throughput"] == 12.5
    assert metrics["success_rate"] == 0.9
    assert metrics["ttft_p99_ms"] == 101
    assert metrics["e2e_avg_ms"] == 100
    assert metrics["itl_p99_ms"] == 11


if __name__ == "__main__":
    test_inference_benchmarker_command_contract()
    test_inference_benchmarker_structured_request_contract()
    test_vllm_bench_command_contract()
    test_metadata_serialization_handles_structured_values()
    test_engine_config_normalizes_devices_and_env()
    test_inference_benchmarker_metrics_contract()
