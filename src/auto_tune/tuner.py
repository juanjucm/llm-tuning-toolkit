import json
import logging
import os
import shutil
import sys
import time
import uuid
from copy import deepcopy
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import coloredlogs
import docker
import yaml
from guidellm.benchmark.profiles import ProfileFactory
from huggingface_hub import HfApi

from auto_tune.display import AutoTuneDisplay
from auto_tune.engine import DockerEngineRuntime, engine_parallel_arguments
from auto_tune.guidellm import GuideLLMError, GuideLLMRunner, metric_names
from auto_tune.slos import evaluate_slos
from auto_tune.tracking import TrackioTracker

coloredlogs.install()

HF_TOKEN = os.getenv("HF_TOKEN", "")
hf_api = HfApi()

# GuideLLM profiles auto-tune declines, mapped to the guidance shown to the user.
# Auto-tune's job is to vary engine parameters against one production load, so a
# profile that measures a baseline or a range of loads belongs in a plain GuideLLM
# run instead.
UNSUPPORTED_LOAD_KINDS = {
    "async": "use 'constant' or 'poisson' with an explicit rate",
    "synchronous": "single-request load measures a latency baseline, not a deployment target; "
    "benchmark it with guidellm directly",
    "sweep": "auto-tune tunes one load level at a time and searches rates itself; "
    "use 'throughput' with scenario.max_rate_finding_attempts",
}



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


class AutoTuner(DockerEngineRuntime):
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
        quiet: bool = False,
    ):
        self.config_path = config_path
        self.config = self._load_config()

        self.root_dir = Path(result_dir or "out")

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
        self.cache_dir = cache_dir or str(Path.home() / ".cache" / "huggingface" / "hub")
        self.display = display
        self.quiet = quiet and display is None

        self.best_throughput = {
            "run_index": None,
            "throughput": 0,
        }
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        # Set up logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(f"{__name__}.AutoTuner.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self._owned_log_handler: Optional[logging.Handler] = None
        if self.display:
            self._owned_log_handler = _DisplayLogHandler(self.display)
            self.logger.addHandler(self._owned_log_handler)
            self.logger.propagate = False
        elif self.quiet:
            # No Rich UI, but the tuner's own records must still reach the terminal:
            # a headless run that reports nothing is indistinguishable from a hung one.
            self._owned_log_handler = logging.StreamHandler(sys.stdout)
            self._owned_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(self._owned_log_handler)
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
            if self._owned_log_handler:
                self.logger.removeHandler(self._owned_log_handler)
                self._owned_log_handler.close()
                self._owned_log_handler = None
                self.logger.propagate = True

    def _load_config(self) -> Dict:
        """
        Load and validate configuration from YAML file
        """
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError("The configuration file must contain a YAML mapping")

        required_keys = ["scenario", "engine", "model", "port", "instance_info"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        scenario = config["scenario"]
        if not isinstance(scenario, dict):
            raise ValueError("scenario must be a mapping")
        if "data" not in scenario:
            # The old prompt/decode descriptors were specific to inference-benchmarker.
            # Failing explicitly prevents a silent benchmark with a different workload.
            raise ValueError(
                "scenario.data is required and must contain one or more GuideLLM data descriptors. "
                "See examples/guidellm-auto-tune.yaml."
            )
        if not isinstance(scenario["data"], list) or not scenario["data"]:
            raise ValueError("scenario.data must be a non-empty list of GuideLLM data descriptors")
        if "goodput_criteria" in scenario and "slos" in scenario:
            raise ValueError("Use scenario.slos; remove the legacy scenario.goodput_criteria key")
        scenario.setdefault("slos", scenario.pop("goodput_criteria", {}))
        slos = scenario["slos"]
        if not isinstance(slos, dict) or not slos:
            raise ValueError(
                "scenario.slos must be a non-empty mapping of min_/max_ SLO names. "
                "See examples/guidellm-auto-tune.yaml."
            )
        # Validate the names here, not in _meets_goodput_criteria: that runs inside the
        # per-config try/except, so a single typo would launch a container and run a full
        # benchmark for every parameter combination before yielding zero results.
        supported = metric_names()
        for name in slos:
            if not name.startswith(("min_", "max_")):
                raise ValueError(f"SLO '{name}' must start with min_ or max_")
            metric = name[4:]
            if metric not in supported:
                raise ValueError(
                    f"SLO '{name}' refers to unsupported metric '{metric}'. "
                    f"Supported metrics: {', '.join(sorted(supported))}"
                )
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
        kind = load["kind"]
        if kind == "throughput":
            load.setdefault("max_concurrency", scenario.get("max_vus", 128))
        elif kind == "concurrent":
            load.setdefault("streams", scenario.get("max_vus", 128))
            # GuideLLM's concurrent profile takes a list of stream counts.
            if not isinstance(load["streams"], list):
                load["streams"] = [load["streams"]]
        elif kind in {"constant", "poisson"}:
            if "rate" not in load:
                raise ValueError(f"scenario.load.rate is required for {kind} workloads")
        elif kind == "replay":
            load.setdefault("time_scale", 1.0)
        elif kind in UNSUPPORTED_LOAD_KINDS:
            raise ValueError(f"scenario.load.kind '{kind}' is not supported: {UNSUPPORTED_LOAD_KINDS[kind]}")
        elif kind not in ProfileFactory.registry:
            # GuideLLM's registry is the only authority on which kinds exist.
            raise ValueError(
                f"scenario.load.kind '{kind}' is not a GuideLLM profile. "
                f"Supported: {', '.join(sorted(ProfileFactory.registry.keys() - UNSUPPORTED_LOAD_KINDS.keys()))}"
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


    def _run_throughput_benchmark(self, run_id: str, output_folder: str) -> Optional[List[Dict]]:
        """Run the production load benchmark and return every load point it measured.

        Args:
            run_id (str): Unique identifier for this benchmark run. Will be used for results file naming.
            output_folder (str): Directory to save benchmark results.
        Returns:
            Optional[List[Dict]]: Normalized metrics per load point, ascending by throughput,
            or None if the benchmark failed.
        """
        self.logger.info("Running %s benchmark...", self.config["scenario"]["load"]["kind"])
        self._set_display_status(f"Running GuideLLM {self.config['scenario']['load']['kind']} benchmark")

        scenario = self.config["scenario"]
        port = self.config["port"]
        output_file = os.path.join(output_folder, f"{scenario['load']['kind']}_{run_id}.json")

        try:
            return self.guidellm.run_all(
                target=f"http://localhost:{port}",
                backend=scenario.get("backend"),
                data=scenario["data"],
                profile=self._primary_profile(),
                constraints=scenario["constraints"],
                output_path=Path(output_file),
                options=scenario.get("guidellm_options"),
                on_output=self._set_guidellm_progress if self.display else None,
                show_console=not self.quiet,
            )
        except GuideLLMError as e:
            self.logger.error(f"Throughput benchmark failed: {e}")
            return None

    def _primary_profile(self) -> Dict:
        """Build the GuideLLM profile for the scenario's production load model."""
        load = self.config["scenario"]["load"]
        return dict(load)

    def _run_rate_benchmark(self, rate: float, run_id: str, output_folder: str) -> Optional[Dict]:
        """
        Run rate benchmark with specific request rate.
        Args:
            rate (float): Request rate in requests per second.
            run_id (str): Unique identifier for this benchmark run. Will be used for results file
            output_folder (str): Directory to save benchmark results.
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
                    "max_concurrency": scenario["load"].get("max_concurrency", scenario.get("max_vus", 128)),
                },
                constraints=scenario["rate_constraints"],
                output_path=Path(output_file),
                options=scenario.get("guidellm_options"),
                on_output=self._set_guidellm_progress if self.display else None,
                show_console=not self.quiet,
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
        return evaluate_slos(self.config["scenario"]["slos"], metrics)

    def _select_load_point(self, candidates: List[Dict]) -> Tuple[Dict, List[Dict], bool]:
        """Choose the load point that represents one engine configuration.

        A concurrent profile measures one point per configured stream count. The
        fastest point that satisfies every SLO is the answer; when none does, report
        the fastest point overall so the configuration's peak throughput is logged.

        Args:
            candidates (List[Dict]): Normalized metrics, one per measured load point.
        Returns:
            Tuple[Dict, List[Dict], bool]: Chosen metrics, its SLO checks, and whether it passed.
        """
        evaluated = [(point, *self._meets_goodput_criteria(point)) for point in candidates]
        qualifying = [entry for entry in evaluated if entry[2]]
        return max(qualifying or evaluated, key=lambda entry: entry[0]["throughput"])

    def _find_optimal_rate(
        self, max_throughput: float, run_id: str, output_folder: str
    ) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """
        Find optimal rate that meets goodput criteria.
        Args:
            max_throughput (float): Maximum throughput from throughput benchmark.
            run_id (str): Unique identifier for this benchmark run. Will be used for results file
            output_folder (str): Directory to save benchmark results.
        Returns:
            Tuple[Dict, List[Dict]]: Metrics at optimal rate and goodput checks, or (None, None) if not found.
        """
        # Start with 90% of max throughput and decrease until goodput is met.
        rate = max_throughput * 0.90
        attempts = 0
        while rate > 0.1 and attempts < self.config["scenario"]["max_rate_finding_attempts"]:
            # Sleep between rate tests to let any pending requests clear
            time.sleep(3)

            metrics = self._run_rate_benchmark(rate=rate, run_id=run_id, output_folder=output_folder)

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
                    parallel_args = engine_parallel_arguments(engine_config)
                    if parallel_args is None:
                        raise ValueError(
                            "engine.value_args_pool.tp-dp-combinations is only supported for "
                            f"vllm and sglang, not '{engine_config.get('kind') or engine_config['name']}'"
                        )
                    combination["value_args"][parallel_args["tp"]] = tp_dp_comb["tp"]
                    combination["value_args"][parallel_args["dp"]] = tp_dp_comb["dp"]

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
        self.logger.info("Starting auto-tune process...")

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
                scenario=self.config["scenario"],
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
                candidates = self._run_throughput_benchmark(
                    run_id=run_id,
                    output_folder=engine_path.as_posix(),
                )
                if not candidates:
                    self.logger.error("Failed to run throughput benchmark, continuing to next config...")
                    continue

                metrics, goodput_checks, meets = self._select_load_point(candidates)

                self.logger.info(f"Throughput: {metrics['throughput']:.2f} req/s")
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
                    )
                    if not metrics:
                        self.logger.info(
                            "Goodput criteria not met. No optimal rate found for this configuration. Continuing..."
                        )
                        self.logger.info(f"{'=' * 60}")
                        continue
                elif not meets:
                    self.logger.info(
                        "SLOs are not met for the configured %s workload; skipping this configuration.",
                        self.config["scenario"]["load"]["kind"],
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
                    if self.display:
                        self.display.set_best(metrics, param_config)
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
        result_summary = {
            "timestamp": self.timestamp,
            "config_file": self.config_path,
            "engine_name": engine_name,
            "instance_info": self.config.get("instance_info", {}),
            "slos": self.config["scenario"]["slos"],
            "scenario": self.config["scenario"],
            "all_results": all_results,
        }
        with open(results_file, "w") as f:
            json.dump(result_summary, f, indent=2)

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
        return result_summary
