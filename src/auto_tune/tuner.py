import json
import logging
import os
import shutil
import tempfile
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
from huggingface_hub import HfApi

from auto_tune.guidellm import GuideLLMError, GuideLLMRunner

coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")
hf_api = HfApi()

class AutoTuner:
    def __init__(
        self,
        config_path: str,
        result_dir: Optional[str] = None,
        dataset_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.config_path = config_path
        self.config = self._load_config()

        if not result_dir:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.root_dir = Path(self._temp_dir.name)
        else:
            self.root_dir = Path(result_dir)

        # Create folder structure for results
        self.results_dir = self.root_dir.joinpath(
            self.config["model"].replace("/", "--"),
            self.config["instance_info"]["gpu_type"],
            self.config["scenario"]["name"],
            "auto-tune",
        )

        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_id = dataset_id
        self.hf_token = hf_token or HF_TOKEN
        self.cache_dir = cache_dir

        self.best_throughput = {
            "run_index": None,
            "throughput": 0,
        }
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        # Docker client
        self.docker_client = docker.from_env()
        self.guidellm = GuideLLMRunner(self.config.get("guidellm", {}).get("command", "guidellm"))

    def _load_config(self) -> Dict:
        """
        Load and validate configuration from YAML file
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        required_keys = ["scenario", "engine", "model", "port", "instance_info"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        scenario = config["scenario"]
        if "data" not in scenario:
            # The old prompt/decode descriptors were specific to inference-benchmarker.
            # Failing explicitly prevents a silent benchmark with a different workload.
            raise ValueError(
                "scenario.data is required and must contain one or more GuideLLM data descriptors. "
                "See examples/guidellm-auto-tune.yaml."
            )
        if not isinstance(scenario["data"], list) or not scenario["data"]:
            raise ValueError("scenario.data must be a non-empty list of GuideLLM data descriptors")
        scenario.setdefault("slos", scenario.pop("goodput_criteria", {}))
        scenario.setdefault("throughput_duration_seconds", 90)
        scenario.setdefault("rate_duration_seconds", 30)
        scenario.setdefault("max_rate_finding_attempts", 3)
        scenario.setdefault("rate_decrease_factor", 0.3)
        # `max_vus` was the former inference-benchmarker name. Keep configs
        # working while using GuideLLM's unambiguous load profile fields.
        load = scenario.setdefault("load", {})
        if not isinstance(load, dict):
            raise ValueError("scenario.load must be a mapping")
        load.setdefault("kind", "throughput")
        if load["kind"] == "throughput":
            load.setdefault("max_concurrency", scenario.get("max_vus", 128))
        elif load["kind"] == "concurrent":
            load.setdefault("streams", scenario.get("max_vus", 128))
        else:
            raise ValueError("scenario.load.kind must be 'throughput' or 'concurrent'")
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

            docker_config = engine_config.get("docker", {})
            docker_args = {
                key: value
                for key, value in docker_config.items()
                if key in {"devices", "group_add", "security_opt"}
            }
            if "devices" in engine_config:
                docker_args["device_requests"] = [
                    docker.types.DeviceRequest(device_ids=engine_config["devices"], capabilities=[["gpu"]])
                ]
            environment = {
                "HF_TOKEN": self.hf_token,
                "HF_HUB_CACHE": "/data/",
                **docker_config.get("environment", {}),
            }

            container = self.docker_client.containers.run(
                image=engine_config["image"],
                command=" ".join([str(a) for a in engine_args]),
                shm_size="2g",
                environment=environment,
                volumes={self.cache_dir: {"bind": "/data/", "mode": "rw"}},
                ports={f"{port}/tcp": port},
                detach=True,
                name=container_name,
                stop_signal="SIGTERM",
                **docker_args,
            )

            return container

        except Exception as e:
            self.logger.error(f"Failed to launch engine: {e}")
            return None

    def _wait_for_server_ready(self, container, port: int, timeout: int = 700) -> bool:
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
                # check if container is still running
                container.reload()
                if container.status != "running":
                    self.logger.error("Engine container has stopped unexpectedly.")
                    return False

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
        """
        Clean up Docker container
        Args:
            container (docker.models.containers.Container): Container to clean up.
        """
        try:
            self.logger.info(f"Stopping container...")
            container.stop(timeout=100)
        except Exception as e:
            self.logger.warning(f"Error stopping container, forcing removal.")

        self.logger.info(f"Waiting for container to exit...")
        try:
            container.reload()
        except Exception as e:
            self.logger.warning(f"Error reloading container status: {e}")
        while container.status != "exited":
            time.sleep(1)
            container.reload()
            self.logger.info(f"Container status: {container.status}")

        self.logger.info(f"Removing container...")
        container.remove(force=True)

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

        try:
            return self.guidellm.run(
                target=f"http://localhost:{port}",
                backend=scenario.get("backend"),
                data=scenario["data"],
                profile=self._primary_profile(),
                duration_seconds=scenario["throughput_duration_seconds"],
                output_path=Path(output_file),
                options=scenario.get("guidellm_options"),
            )
        except GuideLLMError as e:
            self.logger.error(f"Throughput benchmark failed: {e}")
            return None

    def _primary_profile(self) -> Dict:
        """Build the GuideLLM profile for the scenario's production load model."""
        load = self.config["scenario"]["load"]
        if load["kind"] == "throughput":
            return {"kind": "throughput", "max_concurrency": load["max_concurrency"]}
        return {"kind": "concurrent", "streams": load["streams"]}

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

        try:
            return self.guidellm.run(
                target=f"http://localhost:{port}",
                backend=scenario.get("backend"),
                data=scenario["data"],
                profile={
                    "kind": "constant",
                    "rate": rate,
                    "max_concurrency": scenario["load"]["max_concurrency"],
                },
                duration_seconds=scenario["rate_duration_seconds"],
                output_path=Path(output_file),
                options=scenario.get("guidellm_options"),
            )
        except GuideLLMError as e:
            self.logger.error(f"Rate benchmark failed: {e}")
            return None

    def _meets_goodput_criteria(self, metrics: Dict) -> Tuple[List[Dict], bool]:
        """
        Check if metrics meet goodput thresholds. Returns detailed results and overall boolean.

        Args:
            metrics (Dict): Metrics to evaluate.
        Returns:
            Tuple[List[Dict], bool]: List of threshold checks and overall meets status.
        """
        thresholds = self.config["scenario"]["slos"]

        results = []
        for threshold_name, threshold_value in thresholds.items():
            if threshold_name.startswith("max_"):
                metric_name = threshold_name.removeprefix("max_")
                comparison = lambda value: value <= threshold_value
            elif threshold_name.startswith("min_"):
                metric_name = threshold_name.removeprefix("min_")
                comparison = lambda value: value >= threshold_value
            else:
                raise ValueError(f"SLO '{threshold_name}' must start with min_ or max_")
            if metric_name not in metrics:
                raise ValueError(f"SLO '{threshold_name}' refers to unsupported metric '{metric_name}'")
            metric_value = metrics[metric_name]
            results.append({threshold_name: threshold_value, metric_name: metric_value, "meets": comparison(metric_value)})

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

            if not metrics:
                self.logger.warning("Rate benchmark did not produce metrics; trying a lower rate.")
                rate *= 1 - self.config["scenario"]["rate_decrease_factor"]
                attempts += 1
                continue
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
            return [{"value_args": {}, "action_args": {}}]

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

                # Handle special case for tp-dp-combinations
                if "tp-dp-combinations" in combination["value_args"].keys():
                    tp_dp_comb = combination["value_args"].pop("tp-dp-combinations")
                    combination["value_args"]["tensor_parallel_size"] = tp_dp_comb["tp"]
                    combination["value_args"]["data_parallel_size"] = tp_dp_comb["dp"]

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
        autotune_id = uuid.uuid4().hex[:4]
        engine_path = self.results_dir.joinpath(engine_name, f"run_{self.timestamp}_{autotune_id}")
        engine_path.mkdir(parents=True, exist_ok=True)

        param_combinations = self._generate_parameter_combinations()

        # Copy config file to results folder
        shutil.copy2(self.config_path, engine_path / "auto_tune_config.yaml")

        all_results = []
        for i, param_config in enumerate(param_combinations, 1):
            container = None
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"[{i}/{len(param_combinations)}] Testing parameter combination: {param_config}")

            # TODO: add model_name metadata to the run_id.
            run_id = uuid.uuid4().hex[:4]

            # TODO: if throughput is a goodput criteria, only perform throughput benchmark.
            # It makes no sense to do rate finding (decrease rate) if rate is a requirement and is not met by throughput bench.
            try:
                engine_args = self._build_engine_args(param_config)
                container = self._launch_docker_engine(engine_args)
                if not container:
                    continue

                if not self._wait_for_server_ready(container, self.config["port"], self.config["engine"]["timeout"]):
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

                if not meets and self.config["scenario"]["load"]["kind"] == "throughput":
                    # if 90% of throughput is less than best found so far, skip rate finding.
                    # rate finding starts at 90% of throughput.
                    if (metrics["throughput"] * 0.90) <= self.best_throughput["throughput"]:
                        self.logger.info(
                            "Goodput criteria not met, but throughput is worse than best found so far. Skipping rate finding..."
                        )
                        self.logger.info(f"{'=' * 60}")
                        continue

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
                elif not meets:
                    self.logger.info(
                        "SLOs are not met at the configured concurrent stream count; "
                        "skipping this configuration."
                    )
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
                if metrics["throughput"] > self.best_throughput["throughput"]:
                    last_best = self.best_throughput["run_index"]
                    self.best_throughput = {
                        "throughput": metrics["throughput"],
                        "run_index": len(all_results),
                    }
                    self.logger.info(
                        f"NEW BEST CONFIG! Throughput: {self.best_throughput['throughput']:.2f} req/s"
                    )
                    result["is_best"] = True
                    if all_results and last_best is not None:
                        all_results[last_best]["is_best"] = False

                all_results.append(result)
            except Exception as e:
                self.logger.error(f"Error testing parameter config {param_config}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                if container is not None:
                    self._cleanup_container(container)

        # Save all results
        results_file = engine_path / "auto_tune_results.json"
        with open(results_file, "w") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "config_file": self.config_path,
                    "engine_name": engine_name,
                    "instance_info": self.config.get("instance_info", {}),
                    "slos": self.config["scenario"]["slos"],
                    "scenario": self.config["scenario"],
                    "all_results": all_results,
                },
                f,
                indent=2,
            )

        # Upload folder to Huggingface dataset if dataset_id is provided
        if self.dataset_id:
            self.logger.info(f"Uploading results to Huggingface dataset {self.dataset_id}...\n")
            hf_api.upload_folder(
                folder_path=self.results_dir,
                path_in_repo=str(self.results_dir.relative_to(self.root_dir)),
                repo_id=self.dataset_id,
                token=self.hf_token,
                repo_type="dataset",
            )

        # Print summary
        self.logger.info(f"{'=' * 60}")
        self.logger.info("AUTO-TUNE COMPLETE")
        self.logger.info(f"Tested {len(param_combinations)} parameter combinations.")
        self.logger.info(f"Results saved to: {results_file}")
