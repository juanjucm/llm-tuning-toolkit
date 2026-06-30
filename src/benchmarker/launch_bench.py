import argparse
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import coloredlogs
import yaml

from benchmarker.engine import DockerEngineRunner
from benchmarker.engine import cache_mount
from benchmarker.engine import spec_from_engine_config
from benchmarker.engine import token_environment
from benchmarker.executors import BenchmarkExecutionError
from benchmarker.executors import BenchmarkRun
from benchmarker.executors import create_benchmark_executor

coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    coloredlogs.install(level=level, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)


def load_bench_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run configured serving-engine benchmarks with a selectable benchmark backend."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to benchmark YAML config")
    parser.add_argument(
        "--scenarios",
        type=str,
        help='Specific scenarios to run, comma separated (e.g. "balanced,long-context")',
        default="all",
    )
    parser.add_argument(
        "--engines",
        type=str,
        help='Specific engines to test, comma separated (e.g. "vLLM,SGLang")',
        default="all",
    )
    parser.add_argument("--output-path", type=str, help="Directory to save benchmark results", default="./results")
    parser.add_argument(
        "--benchmark-backend",
        default="inference-benchmarker",
        choices=["inference-benchmarker", "vllm-bench"],
        help="Benchmark executor backend. Config benchmark args must match the selected backend.",
    )
    parser.add_argument("--benchmark-command", help="Override benchmark command. Defaults depend on backend.")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "huggingface" / "hub"))
    parser.add_argument("--hf-token", default=HF_TOKEN, help="Hugging Face token for model access")
    parser.add_argument("--show-logs", action="store_true", help="Stream engine container logs while running")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def selected_items(items: List[Dict[str, Any]], requested: str) -> List[Dict[str, Any]]:
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if names == ["all"]:
        return items
    return [item for item in items if item.get("name") in names]


def generate_unique_run_id() -> str:
    return uuid.uuid4().hex[:4]


def stream_container_logs(container_name: str):
    import subprocess

    def log_stream():
        try:
            process = subprocess.Popen(
                ["docker", "logs", "-f", container_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            for line in process.stdout:
                print(f"[{container_name}] {line.rstrip()}")
        except Exception as exc:
            print(f"Error streaming logs for {container_name}: {exc}")

    thread = threading.Thread(target=log_stream, daemon=True)
    thread.start()
    return thread


def benchmark_metadata(
    engine_name: str,
    scenario_name: str,
    scenario_description: str,
    instance_info: Dict[str, Any],
    engine_args: List[Any],
    engine_envs: Dict[str, str],
) -> Dict[str, Any]:
    safe_envs = dict(engine_envs)
    safe_envs.pop("HF_TOKEN", None)
    return {
        "engine": engine_name,
        "instance": instance_info,
        "scenario": scenario_name,
        "scenario_description": scenario_description,
        "engine_args": " ".join(str(arg) for arg in engine_args),
        "engine_envs": safe_envs,
    }


def main():
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    executor = create_benchmark_executor(
        args.benchmark_backend, logger=logger, command=args.benchmark_command, verbose=args.verbose
    )
    engine_runner = DockerEngineRunner(logger)

    try:
        config = load_bench_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
        if not config.get("model"):
            logger.error("No model specified in configuration")
            return 1

        port = config.get("port", 8000)
        instance_info = config.get("instance_info", {})
        scenarios_to_run = selected_items(config.get("scenarios", []), args.scenarios)
        if not scenarios_to_run:
            logger.error("No scenarios to run.")
            return 1

        total_runs = 0
        successful_runs = 0
        failed_runs = 0
        for scenario in scenarios_to_run:
            scenario_name = scenario.get("name", "")
            scenario_description = scenario.get("description", "")
            scenario_bench_args = scenario.get("bench_config", [])
            engines_to_test = selected_items(scenario.get("engines", []), args.engines)
            if not engines_to_test:
                logger.error(f"No engines to run for scenario {scenario_name}.")
                continue

            logger.info(f"Starting scenario: {scenario_name} - {scenario_description or 'No description'}")
            scenario_output_dir = os.path.join(args.output_path, scenario_name, "benchmarking")
            os.makedirs(scenario_output_dir, exist_ok=True)

            for engine in engines_to_test:
                total_runs += 1
                run_id = generate_unique_run_id()
                container = None
                logs_thread = None
                engine_name = engine.get("name", engine.get("image", "engine"))
                extra_env = token_environment(args.hf_token, args.cache_dir)
                spec = spec_from_engine_config(
                    engine,
                    port,
                    extra_env=extra_env,
                    volumes=cache_mount(args.cache_dir),
                )
                try:
                    container = engine_runner.launch(spec, name_prefix="bench")
                    if not container:
                        failed_runs += 1
                        continue
                    if args.show_logs:
                        logs_thread = stream_container_logs(container.name)
                    if not engine_runner.wait_ready(container, port):
                        failed_runs += 1
                        continue

                    output_file = os.path.join(scenario_output_dir, f"{engine_name}_{run_id}.json")
                    bench_args = [*scenario_bench_args, "--run-id", run_id, "--output-path", output_file]
                    executor.run(
                        BenchmarkRun(
                            args=bench_args,
                            extra_meta=benchmark_metadata(
                                engine_name,
                                scenario_name,
                                scenario_description,
                                instance_info,
                                list(spec.command),
                                spec.environment,
                            ),
                        )
                    )
                    successful_runs += 1
                except BenchmarkExecutionError as exc:
                    failed_runs += 1
                    logger.error(f"Benchmark failed for {engine_name} - {scenario_name}: {exc}")
                except Exception as exc:
                    failed_runs += 1
                    logger.exception(f"Error testing {engine_name} in {scenario_name}: {exc}")
                finally:
                    if container:
                        engine_runner.cleanup(container)
                    if logs_thread:
                        logs_thread.join(timeout=5)
                    time.sleep(3)

        logger.info("Benchmark execution completed.")
        logger.info(f"Total runs: {total_runs}, Successful: {successful_runs}, Failed: {failed_runs}")
        return 0 if failed_runs == 0 else 1
    except Exception as exc:
        logger.exception(f"Fatal error: {exc}")
        return 1
