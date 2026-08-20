import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coloredlogs
import docker
import requests
import yaml
from huggingface_hub import HfApi

from auto_tune.display import AutoTuneDisplay
from auto_tune.guidellm import GuideLLMError, GuideLLMRunner
from auto_tune.tracking import TrackioTracker

coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")
hf_api = HfApi()


class _DisplayLogHandler(logging.Handler):
    """Route application logs into the Rich live display."""

    def __init__(self, display: AutoTuneDisplay) -> None:
        super().__init__()
        self.display = display

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.exc_info and record.exc_info[1]:
            error = record.exc_info[1]
            message = f"{message}\n{type(error).__name__}: {error}"
        self.display.log(message, record.levelname)


class AutoTuner:
    def __init__(
        self,
        config_path: str,
        result_dir: Optional[str] = None,
        dataset_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        hf_token: Optional[str] = None,
        display: Optional[AutoTuneDisplay] = None,
        trackio_project: Optional[str] = None,
        trackio_space_id: Optional[str] = None,
        trackio_server_url: Optional[str] = None,
        trackio_group: Optional[str] = None,
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
        self.display = display

        self.best_throughput = {
            "run_index": None,
            "throughput": 0,
        }
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(f"{__name__}.AutoTuner.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self._display_log_handler: Optional[_DisplayLogHandler] = None
        if self.display:
            self._display_log_handler = _DisplayLogHandler(self.display)
            self.logger.addHandler(self._display_log_handler)
            self.logger.propagate = False

        # Docker client
        self.docker_client = docker.from_env()
        self.guidellm = GuideLLMRunner(self.config.get("guidellm", {}).get("command", "guidellm"))
        if trackio_space_id and trackio_server_url:
            raise ValueError("trackio_space_id and trackio_server_url cannot be used together")
        if not trackio_project and any((trackio_space_id, trackio_server_url, trackio_group)):
            raise ValueError("trackio_project is required when other Trackio options are provided")
        trackio_config = None
        if trackio_project:
            trackio_config = {"project": trackio_project}
            if trackio_space_id:
                trackio_config["space_id"] = trackio_space_id
            if trackio_server_url:
                trackio_config["server_url"] = trackio_server_url
            if trackio_group:
                trackio_config["group"] = trackio_group
        self.tracker = TrackioTracker(trackio_config, self.logger, hf_token=self.hf_token)

    def _set_display_status(self, status: str) -> None:
        display = getattr(self, "display", None)
        if display:
            display.set_status(status)

    def _set_guidellm_progress(self, progress: str) -> None:
        display = getattr(self, "display", None)
        if display:
            display.set_guidellm_progress(progress)

    def _close_display(self) -> None:
        """Stop the live view and detach its logger handler."""
        try:
            if self.display:
                self.display.stop()
        finally:
            if self._display_log_handler:
                self.logger.removeHandler(self._display_log_handler)
                self._display_log_handler.close()
                self._display_log_handler = None
                self.logger.propagate = True

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
        has_legacy_throughput_duration = "throughput_duration_seconds" in scenario
        has_legacy_rate_duration = "rate_duration_seconds" in scenario
        legacy_throughput_duration = scenario.get("throughput_duration_seconds", 90)
        legacy_rate_duration = scenario.get("rate_duration_seconds", 30)
        default_max_requests = scenario.setdefault("max_requests", 1000)
        default_rate_max_requests = scenario.setdefault("rate_max_requests", default_max_requests)
        has_primary_constraints = "constraints" in scenario
        scenario["constraints"] = self._normalize_constraints(
            scenario.get("constraints"),
            legacy_throughput_duration,
            "scenario.constraints",
            default=(
                [{"kind": "max_duration", "seconds": legacy_throughput_duration}]
                if has_legacy_throughput_duration
                else [{"kind": "max_requests", "count": default_max_requests}]
            ),
        )
        scenario["rate_constraints"] = self._normalize_constraints(
            scenario.get("rate_constraints"),
            legacy_rate_duration,
            "scenario.rate_constraints",
            default=(
                deepcopy(scenario["constraints"])
                if has_primary_constraints
                else (
                    [{"kind": "max_duration", "seconds": legacy_rate_duration}]
                    if has_legacy_rate_duration
                    else [{"kind": "max_requests", "count": default_rate_max_requests}]
                )
            ),
        )
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
        elif load["kind"] in {"constant", "poisson"}:
            if "rate" not in load:
                raise ValueError(f"scenario.load.rate is required for {load['kind']} workloads")
        elif load["kind"] == "replay":
            load.setdefault("time_scale", 1.0)
        else:
            raise ValueError(
                "scenario.load.kind must be 'throughput', 'concurrent', 'constant', 'poisson', or 'replay'"
            )
        arguments = scenario.get("guidellm_options", {}).get("arguments", {})
        if "constraint" in arguments or "constraints" in arguments:
            raise ValueError("Use scenario.constraints or scenario.rate_constraints, not guidellm_options.arguments.constraint")
        if "trackio" in config:
            raise ValueError("Trackio is configured through the auto-tune CLI; remove 'trackio' from the YAML file")
        return config

    @staticmethod
    def _normalize_constraints(
        value: object,
        legacy_duration: int | float,
        name: str,
        *,
        default: list[Dict] | None = None,
    ) -> list[Dict | str]:
        """Normalize GuideLLM constraint descriptors and preserve old duration settings."""
        if value is None:
            value = default if default is not None else [{"kind": "max_duration", "seconds": legacy_duration}]
        if isinstance(value, (dict, str)):
            value = [value]
        if not isinstance(value, list) or not value:
            raise ValueError(f"{name} must be a non-empty GuideLLM constraint list")
        if not all(isinstance(item, (dict, str)) for item in value):
            raise ValueError(f"{name} items must be GuideLLM constraint mappings or descriptors")
        return value

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
        self._set_display_status("Waiting for engine health check")
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
                    self._set_display_status("Engine ready")
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
        self._set_display_status("Stopping engine")
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
        self.logger.info("Running %s benchmark...", self.config["scenario"]["load"]["kind"])
        self._set_display_status(f"Running GuideLLM {self.config['scenario']['load']['kind']} benchmark")

        scenario = self.config["scenario"]
        port = self.config["port"]
        output_file = os.path.join(output_folder, f"{scenario['load']['kind']}_{run_id}.json")

        try:
            return self.guidellm.run(
                target=f"http://localhost:{port}",
                backend=scenario.get("backend"),
                data=scenario["data"],
                profile=self._primary_profile(),
                constraints=scenario["constraints"],
                output_path=Path(output_file),
                options=scenario.get("guidellm_options"),
                on_output=self._set_guidellm_progress if self.display else None,
            )
        except GuideLLMError as e:
            self.logger.error(f"Throughput benchmark failed: {e}")
            return None

    def _primary_profile(self) -> Dict:
        """Build the GuideLLM profile for the scenario's production load model."""
        load = self.config["scenario"]["load"]
        return dict(load)

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
        self._set_display_status(f"Running GuideLLM rate benchmark at {rate:.2f} req/s")

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
                constraints=scenario["rate_constraints"],
                output_path=Path(output_file),
                options=scenario.get("guidellm_options"),
                on_output=self._set_guidellm_progress if self.display else None,
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
            self.tracker.log_metrics(metrics, slo_met=meets, target_rate=rate)
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
        """Run auto-tuning and always release terminal display resources."""
        try:
            return self._run_auto_tune()
        finally:
            self._close_display()

    def _run_auto_tune(self) -> Dict:
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
        if self.display:
            self.display.start(
                total_configs=len(param_combinations),
                scenario_name=self.config["scenario"]["name"],
                model=self.config["model"],
            )

        # Copy config file to results folder
        shutil.copy2(self.config_path, engine_path / "auto_tune_config.yaml")

        all_results = []
        for i, param_config in enumerate(param_combinations, 1):
            container = None
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"[{i}/{len(param_combinations)}] Testing parameter combination: {param_config}")
            if self.display:
                self.display.begin_config(i, len(param_combinations), param_config)

            # TODO: add model_name metadata to the run_id.
            run_id = uuid.uuid4().hex[:4]

            # TODO: if throughput is a goodput criteria, only perform throughput benchmark.
            # It makes no sense to do rate finding (decrease rate) if rate is a requirement and is not met by throughput bench.
            try:
                engine_args = self._build_engine_args(param_config)
                container = self._launch_docker_engine(engine_args)
                if not container:
                    continue

                if not self._wait_for_server_ready(
                    container,
                    self.config["port"],
                    self.config["engine"].get("timeout", 700),
                ):
                    self.logger.error("Server failed to start properly")
                    continue

                self.tracker.start_run(
                    name=f"{engine_name}-{run_id}",
                    group=self.config["scenario"]["name"],
                    config={
                        "model": self.config["model"],
                        "scenario": self.config["scenario"]["name"],
                        "load": self.config["scenario"]["load"],
                        "engine": engine_name,
                        "engine_image": self.config["engine"]["image"],
                        "engine_args": engine_args,
                        "parameters": param_config,
                        "instance_info": self.config.get("instance_info", {}),
                    },
                )
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
                self.tracker.log_metrics(metrics, slo_met=meets)
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
                    self.logger.info("SLOs are not met for the configured %s workload; skipping this configuration.",
                                     self.config["scenario"]["load"]["kind"])
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
                    if self.display:
                        self.display.set_best(metrics["throughput"])
                    self.logger.info(
                        f"NEW BEST CONFIG! Throughput: {self.best_throughput['throughput']:.2f} req/s"
                    )
                    result["is_best"] = True
                    if all_results and last_best is not None:
                        all_results[last_best]["is_best"] = False

                all_results.append(result)
            except Exception as e:
                self.logger.exception("Error testing parameter config %s: %s", param_config, e)
            finally:
                self.tracker.finish_run()
                try:
                    if container is not None:
                        self._cleanup_container(container)
                finally:
                    if self.display:
                        self.display.complete_config(i)

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
        self._set_display_status("Complete")
