import json
import logging
import os
import subprocess
import time
import uuid
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coloredlogs
import docker
import requests
import yaml

coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK_TOOL_CMD = "inference-benchmarker"


class AutoTuner:
    def __init__(self, config_path: str, result_dir: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.results_dir = Path(result_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.best_throughput = 0

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Docker client
        self.docker_client = docker.from_env()

    def _load_config(self) -> Dict:
        """
        Load and validate configuration from YAML file
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = ["scenario", "engine"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        return config

    def _build_engine_args(self, param_config: Dict) -> List[str]:
        """
        Build engine arguments from configuration.

        Args:
            param_config (Dict): Dictionary with 'value_args' and 'action_args'.
        Returns:
            List[str]: List of command-line arguments for the engine.
        """
        args = list(self.config["engine"]["base_args"])

        # Handle value arguments (--param value)
        for param, value in param_config.get("value_args", {}).items():
            args.extend([f"--{param.replace('_', '-')}", str(value)])

        # Handle action arguments (boolean flags)
        for param, value in param_config.get("action_args", {}).items():
            if value:
                args.append(f"--{param.replace('_', '-')}")

        return args

    def _launch_docker_engine(self, engine_args: List[str]) -> Optional[docker.models.containers.Container]:
        """
        Launch a Docker container running the engine.
        Args:
            engine_args (List[str]): List of engine arguments.
        Returns:
            Optional[docker.models.containers.Container]: Docker container instance or None if failed.
        """
        try:
            engine_config = self.config["engine"]
            port = self.config["port"]

            container_name = f"autotune_engine_{int(time.time())}"
            self.logger.info(f"Starting engine container: {container_name}")

            device_requests = [docker.types.DeviceRequest(device_ids=["all"], capabilities=[["gpu"]])]

            container = self.docker_client.containers.run(
                image=engine_config["image"],
                command=" ".join([str(a) for a in engine_args]),
                environment={"HF_TOKEN": HF_TOKEN},
                ports={f"{port}/tcp": port},
                detach=True,
                name=container_name,
                remove=False,
                auto_remove=False,
                device_requests=device_requests,
            )

            return container

        except Exception as e:
            self.logger.error(f"Failed to launch engine: {e}")
            return None

    def _wait_for_server_ready(self, port: int, timeout: int = 300) -> bool:
        """
        Wait for the server to be ready to accept requests.
        Args:
            port (int): Port number where the server is expected to be listening.
            timeout (int): Maximum time to wait in seconds.
        Returns:
            bool: True if server is ready, False if timeout occurs.
        """
        self.logger.info("Waiting for engine to be ready...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=5)
                if response.status_code == 200:
                    self.logger.info("Engine is ready!")
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)

        self.logger.error("Timeout waiting for engine to be ready")
        return False

    def _cleanup_container(self, container: docker.models.containers.Container):
        """Clean up Docker container"""
        try:
            container.stop(timeout=20)
            container.remove()
            self.logger.info(f"Cleaned up container: {container.name}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup container {container.name}: {e}")

    def _run_inference_benchmarker(self, bench_args: List[str]):
        """
        Run the benchmark command using inference-benchmarker tool

        Args:
            bench_args (List[str]): List of arguments for the benchmark tool.
        Raises:
            subprocess.CalledProcessError: If the benchmark command fails.
        """
        try:
            cmd = [BENCHMARK_TOOL_CMD]
            cmd.extend(bench_args)
            cmd.extend(["--no-console"])  # disable UI

            cmd = [str(c) for c in cmd]

            self.logger.info(f"Running benchmark: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=None, check=True)

            # self.logger.info("=== SUBPROCESS OUTPUT ===")
            # self.logger.info(f"Return code: {proc.returncode}")
            # self.logger.info(f"STDOUT:\n{proc.stdout}")
            # if proc.stderr:
            #     self.logger.info(f"STDERR:\n{proc.stderr}")
            self.logger.info("Benchmark completed successfully")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Benchmark failed with return code {e.returncode}: {e}")
            self.logger.error(f"Command: {' '.join(cmd)}")
            self.logger.error(f"STDOUT:\n{e.stdout if e.stdout else 'None'}")
            self.logger.error(f"STDERR:\n{e.stderr if e.stderr else 'None'}")
            
            # Print full traceback for debugging
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e
        except Exception as e:
            self.logger.error(f"Unexpected error running benchmark: {e}")
            import traceback
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise e

    def _get_metrics_from_results(self, results_dict: Dict) -> Dict:
        """
        Extract key metrics from benchmark results.
        Results format expected to match inference-benchmarker output. Only one result should be present
        since AutoTuner executes one-run throughput/rate benchmarks.

        Args:
            results_dict (Dict): Parsed JSON results from benchmark.
        Returns:
            Dict: Extracted metrics including throughput, latencies, success rates, etc.
        """
        metrics = {}
        results = results_dict.get("results", [])
        result = (
            results[-1] if results else {}
        )  # NOTE: following inference-benchmarker output logs format, take last result (skip warmup)

        # Throughput and success metrics
        metrics["throughput"] = result.get("request_rate", 0)
        metrics["total_requests"] = result.get("total_requests", 0)
        metrics["successful_requests"] = result.get("successful_requests", 0)
        metrics["failed_requests"] = result.get("failed_requests", 0)
        metrics["success_rate"] = result.get("successful_requests", 0) / max(result.get("total_requests", 1), 1)

        # Latency metrics
        ttft = result.get("time_to_first_token_ms", {})
        e2e = result.get("e2e_latency_ms", {})
        itl = result.get("inter_token_latency_ms", {})
        metrics.update(
            {
                "ttft_p99_ms": ttft.get("p99", float("inf")),
                "ttft_p95_ms": ttft.get("p95", float("inf")),
                "ttft_p90_ms": ttft.get("p90", float("inf")),
                "ttft_p80_ms": ttft.get("p80", float("inf")),
                "ttft_p70_ms": ttft.get("p70", float("inf")),
                "ttft_p60_ms": ttft.get("p60", float("inf")),
                "ttft_p50_ms": ttft.get("p50", float("inf")),
                "ttft_avg_ms": ttft.get("avg", float("inf")),
                "e2e_p99_ms": e2e.get("p99", float("inf")),
                "e2e_p95_ms": e2e.get("p95", float("inf")),
                "e2e_p90_ms": e2e.get("p90", float("inf")),
                "e2e_p80_ms": e2e.get("p80", float("inf")),
                "e2e_p70_ms": e2e.get("p70", float("inf")),
                "e2e_p60_ms": e2e.get("p60", float("inf")),
                "e2e_p50_ms": e2e.get("p50", float("inf")),
                "e2e_avg_ms": e2e.get("avg", float("inf")),
                "itl_p99_ms": itl.get("p99", float("inf")),
                "itl_p95_ms": itl.get("p95", float("inf")),
                "itl_p90_ms": itl.get("p90", float("inf")),
                "itl_p80_ms": itl.get("p80", float("inf")),
                "itl_p70_ms": itl.get("p70", float("inf")),
                "itl_p60_ms": itl.get("p60", float("inf")),
                "itl_p50_ms": itl.get("p50", float("inf")),
                "itl_avg_ms": itl.get("avg", float("inf")),
            }
        )

        return metrics

    def _run_throughput_benchmark(self, run_id: str, output_folder: str, engine_config: str) -> Optional[Dict]:
        """Run throughput benchmark to discover maximum throughput

        Args:
            run_id (str): Unique identifier for this benchmark run. Will be used for results file naming.
            output_folder (str): Directory to save benchmark results.
            engine_config (str): Current docker engine command.
        Returns:
            Optional[Dict]: Parsed benchmark results or None if failed.
        """
        self.logger.info("Running throughput benchmark...")

        scenario = self.config["scenario"]
        port = self.config["port"]
        output_file = os.path.join(output_folder, f"throughput_{run_id}.json")

        # Build benchmark arguments for throughput test
        bench_args = [
            "--url",
            f"http://localhost:{port}",
            "--benchmark-kind",
            "throughput",
            "--max-vus",
            str(scenario["max_vus"]),
            "--duration",
            scenario["throughput_duration"],
            "--prompt-options",
            scenario["prompt_options"],
            "--decode-options",
            scenario["decode_options"],
            "--tokenizer-name",
            self.config["model"],
            "--output-path",
            output_file,
            "--run-id",
            run_id,
        ]

        if scenario.get("dataset_file"):
            bench_args.extend(["--dataset-file", scenario["dataset_file"]])

        metadata = (
            f"autotune=true,engine_name={self.config['engine']['name']},docker_engine_args={engine_config}"
        )
        bench_args.extend(["--extra-meta", metadata])

        try:
            self._run_inference_benchmarker(bench_args)
        except Exception as e:
            self.logger.error(f"Throughput benchmark failed: {e}")
            return None

        # return metrics
        metrics = {}
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                results = json.load(f)
                metrics = self._get_metrics_from_results(results)
                if metrics:
                    return metrics
                else:
                    self.logger.error("Failed to extract metrics from results.")
                    return None
        else:
            self.logger.error("Results file not found after benchmark")
            return None

    def _run_rate_benchmark(
        self, rate: float, run_id: str, output_folder: str, engine_config: str
    ) -> Optional[Dict]:
        """
        Run rate benchmark with specific request rate.
        Args:
            rate (float): Request rate in requests per second.
            run_id (str): Unique identifier for this benchmark run. Will be used for results file
            output_folder (str): Directory to save benchmark results.
            engine_config (str): Current docker engine command.
        Returns:
            Optional[Dict]: Parsed benchmark results or None if failed.
        """
        self.logger.info(f"Running rate benchmark at {rate:.2f} req/s")

        scenario = self.config["scenario"]
        port = self.config["port"]
        output_file = os.path.join(output_folder, f"rate_@{rate:.2f}_{run_id}.json")

        # Build benchmark arguments for rate test
        bench_args = [
            "--url",
            f"http://localhost:{port}",
            "--benchmark-kind",
            "rate",
            "--max-vus",
            str(scenario["max_vus"]),
            "--duration",
            str(scenario["rate_duration"]),
            "--rates",
            str(rate),
            "--prompt-options",
            scenario["prompt_options"],
            "--decode-options",
            scenario["decode_options"],
            "--tokenizer-name",
            self.config["model"],
            "--output-path",
            output_file,
            "--run-id",
            run_id,
        ]

        if scenario.get("dataset_file"):
            bench_args.extend(["--dataset-file", scenario["dataset_file"]])

        metadata = (
            f"autotune=true,engine_name={self.config['engine']['name']},docker_engine_args={engine_config}"
        )
        bench_args.extend(["--extra-meta", metadata])

        try:
            self._run_inference_benchmarker(bench_args)
        except Exception as e:
            self.logger.error(f"Rate benchmark failed: {e}")
            return None

        # return metrics
        metrics = {}
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                results = json.load(f)
                metrics = self._get_metrics_from_results(results)
                if metrics:
                    return metrics
                else:
                    self.logger.error("Failed to extract metrics from results.")
                    return None
        else:
            self.logger.error("Results file not found after benchmark")
            return None

    def _meets_goodput_criteria(self, metrics: Dict) -> Tuple[List[Dict], bool]:
        """
        Check if metrics meet goodput thresholds. Returns detailed results and overall boolean.

        Args:
            metrics (Dict): Metrics to evaluate.
        Returns:
            Tuple[List[Dict], bool]: List of threshold checks and overall meets status.
        """
        thresholds = self.config["scenario"]["goodput_criteria"]

        # for each metric in metrics, check if a threshold exists and check if reached
        results = []
        for metric_name, metric_value in metrics.items():
            for threshold_name, threshold_value in thresholds.items():
                if metric_name in threshold_name:
                    if ("max" in threshold_name and metric_value > threshold_value) or (
                        "min" in threshold_name and metric_value < threshold_value
                    ):
                        results.append(
                            {threshold_name: threshold_value, metric_name: metric_value, "meets": False}
                        )
                    else:
                        results.append(
                            {threshold_name: threshold_value, metric_name: metric_value, "meets": True}
                        )

        meets = all(r["meets"] for r in results)

        return results, meets

    def _find_optimal_rate(
        self, max_throughput: float, run_id: str, output_folder: str, engine_config: Dict
    ) -> Tuple[Dict, List[Dict]]:
        """
        Find optimal rate that meets goodput criteria.
        Args:
            max_throughput (float): Maximum throughput from throughput benchmark.
            run_id (str): Unique identifier for this benchmark run. Will be used for results file
            output_folder (str): Directory to save benchmark results.
            engine_config (str): Current docker engine command.
        Returns:
            Tuple[Dict, List[Dict]]: Metrics at optimal rate and goodput checks, or (None, None) if not found.
        """
        # Start with 90% of max throughput and decrease until goodput is met.
        rate = max_throughput * 0.90
        attempts = 0
        while rate > 0.1 and attempts < self.config["scenario"]["max_rate_finding_attempts"]:
            # Sleep between rate tests to let any pending requests clear
            time.sleep(3)

            metrics = self._run_rate_benchmark(
                rate=rate, run_id=run_id, output_folder=output_folder, engine_config=engine_config
            )

            self.logger.info(f"Throughput: {metrics['throughput']:.2f} req/s")

            goodput_checks, meets = self._meets_goodput_criteria(metrics)
            self.logger.info("Goodput SLOs checks:")
            self.logger.info(json.dumps(goodput_checks, indent=2))

            if meets:
                self.logger.info(f"Found optimal rate: {rate:.2f} req/s")

                return metrics, goodput_checks
            else:
                self.logger.info("Goodput criteria not met, finding optimal rate for this config...")
                self.logger.info(f"{'=' * 60}")
                rate *= 1 - self.config["scenario"]["rate_decrease_factor"]
                attempts += 1

        return None, None

    def _generate_parameter_combinations(self) -> List[Dict]:
        """
        Generate all parameter combinations to test.
        """
        engine_config = self.config["engine"]
        value_args_pool = engine_config.get("value_args_pool", {})
        action_args_pool = engine_config.get("action_args_pool", {})

        # Get parameter names and values for each type
        if not value_args_pool and not action_args_pool:
            self.logger.warning("No tunable parameters defined in config, only base args will be used.")
            return []

        value_param_names = []
        value_param_values = []
        action_param_names = []
        action_param_values = []
        if value_args_pool:
            value_param_names = list(value_args_pool.keys())
            value_param_values = [value_args_pool[name] for name in value_param_names]
        if action_args_pool:
            action_param_names = list(action_args_pool.keys())
            action_param_values = [action_args_pool[name] for name in action_param_names]

        # Generate all combinations
        combinations = []

        # Generate combinations for value params (or empty if none)
        value_combinations = list(product(*value_param_values)) if value_param_values else [()]
        action_combinations = list(product(*action_param_values)) if action_param_values else [()]

        for value_combo in value_combinations:
            for action_combo in action_combinations:
                combination = {
                    "value_args": dict(zip(value_param_names, value_combo)) if value_param_names else {},
                    "action_args": dict(zip(action_param_names, action_combo)) if action_param_names else {},
                }
                combinations.append(combination)

        self.logger.info(
            f"Generated {len(combinations)} parameter combinations to test for engine {engine_config['name']}."
        )

        return combinations

    def run_auto_tune(self) -> Dict:
        """
        Run the auto-tuning process.
        """
        # TODO: implement verbose/normal logging levels.
        self.logger.info(f"Starting auto-tune process...")

        # TODO: extend to support multiple engine auto-tuning.
        engine_name = self.config["engine"]["name"]
        run_id = uuid.uuid4().hex[:4]
        engine_path = self.results_dir.joinpath(engine_name, f"run_{run_id}_{self.timestamp}")
        engine_path.mkdir(parents=True, exist_ok=True)

        param_combinations = self._generate_parameter_combinations()

        all_results = []
        for i, param_config in enumerate(param_combinations, 1):
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"[{i}/{len(param_combinations)}] Testing parameter combination: {param_config}")

            run_id = uuid.uuid4().hex[:4]

            # TODO: if throughput is a goodput criteria, only perform throughput benchmark.
            # It makes no sense to do rate finding (decrease rate) if rate is a requirement and is not met by throughput bench.
            try:
                # TODO: save logs to debug if needed
                engine_args = self._build_engine_args(param_config)
                container = self._launch_docker_engine(engine_args)
                if not container:
                    continue

                if not self._wait_for_server_ready(self.config["port"]):
                    self.logger.error("Server failed to start properly")
                    continue

                metrics = self._run_throughput_benchmark(
                    run_id=run_id,
                    output_folder=engine_path.as_posix(),
                    engine_config=" ".join([str(a) for a in engine_args]),
                )
                if not metrics:
                    self.logger.error("Failed to run throughput benchmark, continuing to next config...")
                    continue

                self.logger.info(f"Throughput: {metrics['throughput']:.2f} req/s")

                goodput_checks, meets = self._meets_goodput_criteria(metrics)
                self.logger.info("Goodput SLOs checks:")
                self.logger.info(json.dumps(goodput_checks, indent=2))
                if not meets:
                    self.logger.info("Goodput criteria not met, finding optimal rate for this config...")
                    self.logger.info(f"{'=' * 60}")
                    metrics, goodput_checks = self._find_optimal_rate(
                        metrics["throughput"],
                        run_id=run_id,
                        output_folder=engine_path.as_posix(),
                        engine_config=" ".join([str(a) for a in engine_args]),
                    )
                    if not metrics:
                        self.logger.info(
                            "Goodput criteria not met. No optimal rate found for this configuration. Continuing..."
                        )
                        self.logger.info(f"{'=' * 60}")
                        continue

                self.logger.info(
                    f"Goodput criteria met! Max throughput: {metrics['throughput']:.2f} req/s for this configuration."
                )
                self.logger.info(f"{'=' * 60}")

                result = {
                    "run_id": run_id,
                    "tunable_parameters_config": param_config,
                    "engine_container_command": engine_args,
                    "metrics": metrics,
                    "goodput_checks": goodput_checks,
                    "is_best": False,
                }
                if metrics["throughput"] > self.best_throughput:
                    self.best_throughput = metrics["throughput"]
                    self.logger.info(f"NEW BEST CONFIG! Throughput: {self.best_throughput:.2f} req/s")
                    result["is_best"] = True
                    if all_results:
                        all_results[-1]["is_best"] = False

                all_results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing parameter config {param_config}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if container:
                    self._cleanup_container(container)

        # Save all results
        results_file = engine_path / "auto_tune_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "config_file": self.config_path,
                    "goodput_criteria": self.config["scenario"]["goodput_criteria"],
                    "scenario": self.config["scenario"],
                    "all_results": all_results,
                },
                f,
                indent=2,
            )

        # Print summary
        self.logger.info(f"{'=' * 60}")
        self.logger.info("AUTO-TUNE COMPLETE")
        self.logger.info(f"Tested {len(param_combinations)} parameter combinations.")
        self.logger.info(f"Results saved to: {results_file}")
