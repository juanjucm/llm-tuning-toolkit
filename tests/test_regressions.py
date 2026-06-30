import importlib
import logging
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def install_stubs():
    sys.modules.setdefault("coloredlogs", types.SimpleNamespace(install=lambda *args, **kwargs: None))
    sys.modules.setdefault(
        "requests",
        types.SimpleNamespace(RequestException=Exception, get=lambda *args, **kwargs: None),
    )
    sys.modules.setdefault("yaml", types.SimpleNamespace(safe_load=lambda stream: {}))

    hf_module = types.ModuleType("huggingface_hub")
    hf_module.HfApi = lambda: types.SimpleNamespace(upload_folder=lambda **kwargs: None)
    sys.modules.setdefault("huggingface_hub", hf_module)

    docker_module = types.ModuleType("docker")
    docker_models = types.ModuleType("docker.models")
    docker_containers = types.ModuleType("docker.models.containers")
    docker_containers.Container = object
    docker_models.containers = docker_containers
    docker_module.models = docker_models
    docker_module.types = types.SimpleNamespace(DeviceRequest=lambda **kwargs: kwargs)
    docker_module.from_env = lambda: None
    sys.modules.setdefault("docker", docker_module)
    sys.modules.setdefault("docker.models", docker_models)
    sys.modules.setdefault("docker.models.containers", docker_containers)


def make_tuner(config):
    tuner_module = importlib.import_module("auto_tune.tuner")
    tuner = tuner_module.AutoTuner.__new__(tuner_module.AutoTuner)
    tuner.config = config
    tuner.logger = logging.getLogger("test")
    return tuner


def test_base_config_runs_without_tunable_pool():
    install_stubs()
    tuner = make_tuner({"engine": {"name": "vllm"}, "scenario": {}})
    assert tuner._generate_parameter_combinations() == [{"value_args": {}, "action_args": {}}]


def test_goodput_criteria_are_exact():
    install_stubs()
    tuner = make_tuner(
        {"scenario": {"goodput_criteria": {"min_success_rate": 0.9, "max_ttft_p99_ms": 100}}}
    )
    checks, meets = tuner._meets_goodput_criteria({"success_rate": 0.95, "ttft_p99_ms": 80})
    assert meets
    assert len(checks) == 2

    tuner = make_tuner({"scenario": {"goodput_criteria": {"min_success": 0.9}}})
    try:
        tuner._meets_goodput_criteria({"success_rate": 0.95})
    except ValueError as exc:
        assert "min_success" in str(exc)
    else:
        raise AssertionError("unknown goodput criterion should fail")


def test_run_benchmark_reports_failure():
    install_stubs()
    bench = importlib.import_module("benchmarker.launch_bench")
    original_run = bench.subprocess.run

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(2, args[0], output="out", stderr="err")

    try:
        bench.subprocess.run = fail
        assert not bench.run_benchmark(
            scenario_name="s",
            scenario_description="",
            instance_info="",
            bench_args=[],
            engine_name="e",
            engine_args="",
            engine_envs={},
            run_id="r",
            output_path="/tmp/out.json",
            logger=logging.getLogger("test"),
        )
    finally:
        bench.subprocess.run = original_run


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
