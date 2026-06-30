import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import coloredlogs
import yaml
from huggingface_hub import HfApi

from benchmarker.engine import DockerEngineRunner
from benchmarker.engine import cache_mount
from benchmarker.engine import spec_from_engine_config
from benchmarker.engine import token_environment
from benchmarker.executors import BenchmarkExecutionError
from benchmarker.executors import BenchmarkRequest
from benchmarker.executors import BenchmarkRun
from benchmarker.executors import create_benchmark_executor

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
        benchmark_backend: str = "inference-benchmarker",
        benchmark_command: Optional[str] = None,
        verbose: bool = False,
    ):
        self.config_path = config_path
        self.config = self._load_config()
        self.root_dir = Path(result_dir or "auto_tune_results")
        self.results_dir = self.root_dir.joinpath(
            self.config["model"].replace("/", "--"),
            self.config["instance_info"]["gpu_type"],
            self.config["scenario"]["name"],
            "auto-tune",
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)

        log_level = logging.DEBUG if verbose else logging.INFO
        coloredlogs.install(level=log_level, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

        self.dataset_id = dataset_id
        self.hf_token = hf_token or HF_TOKEN
        self.cache_dir = cache_dir
        self.best_throughput = {"run_index": None, "throughput": 0}
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.engine_runner = DockerEngineRunner(self.logger)
        self.benchmark_executor = create_benchmark_executor(
            benchmark_backend, logger=self.logger, command=benchmark_command, verbose=verbose
        )

    def _load_config(self) -> Dict:
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        for key in ["model", "port", "instance_info", "scenario", "engine"]:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        return config

    def _build_engine_args(self, param_config: Dict) -> List[str]:
        args = list(self.config["engine"]["base_args"])
        for param, value in param_config.get("value_args", {}).items():
            args.extend([f"--{param.replace('_', '-')}", str(value)])
        for param, value in param_config.get("action_args", {}).items():
            if value:
                args.append(f"--{param.replace('_', '-')}")
        return args

    def _engine_spec(self, engine_args: List[str]):
        engine_config = dict(self.config["engine"])
        engine_config["args"] = engine_args
        env = token_environment(self.hf_token, self.cache_dir)
        return spec_from_engine_config(
            engine_config,
            self.config["port"],
            extra_env=env,
            volumes=cache_mount(self.cache_dir),
        )

    def _get_metrics_from_results(self, results_dict: Dict) -> Dict:
        return self.benchmark_executor.extract_metrics(results_dict)

    def _run_benchmark(self, run: BenchmarkRun) -> bool:
        try:
            self.benchmark_executor.run(run)
            return True
        except BenchmarkExecutionError as exc:
            self.logger.error(f"Benchmark failed: {exc}")
            return False

    def _run_benchmark_request(self, request: BenchmarkRequest) -> bool:
        try:
            run = self.benchmark_executor.build_request_run(request)
        except (NotImplementedError, ValueError) as exc:
            self.logger.error(f"Benchmark backend cannot run this request: {exc}")
            return False
        return self._run_benchmark(run)

    def _run_throughput_benchmark(self, run_id: str, output_folder: str, engine_config: str) -> Optional[Dict]:
        self.logger.info("Running throughput benchmark...")
        scenario = self.config["scenario"]
        port = self.config["port"]
        output_file = os.path.join(output_folder, f"throughput_{run_id}.json")
        request = BenchmarkRequest(
            kind="throughput",
            url=f"http://localhost:{port}",
            max_vus=scenario["max_vus"],
            duration=scenario["throughput_duration"],
            tokenizer_name=self.config["model"],
            output_path=output_file,
            run_id=run_id,
            prompt_options=scenario.get("prompt_options"),
            decode_options=scenario.get("decode_options"),
            dataset_file=scenario.get("dataset_file"),
            extra_meta=self._metadata(engine_config),
        )

        if not self._run_benchmark_request(request):
            return None
        return self._load_metrics(output_file)

    def _run_rate_benchmark(
        self, rate: float, run_id: str, output_folder: str, engine_config: str
    ) -> Optional[Dict]:
        self.logger.info(f"Running rate benchmark at {rate:.2f} req/s")
        scenario = self.config["scenario"]
        port = self.config["port"]
        output_file = os.path.join(output_folder, f"rate_@{rate:.2f}_{run_id}.json")
        request = BenchmarkRequest(
            kind="rate",
            url=f"http://localhost:{port}",
            max_vus=scenario["max_vus"],
            duration=str(scenario["rate_duration"]),
            tokenizer_name=self.config["model"],
            output_path=output_file,
            run_id=run_id,
            prompt_options=scenario.get("prompt_options"),
            decode_options=scenario.get("decode_options"),
            dataset_file=scenario.get("dataset_file"),
            rate=rate,
            extra_meta=self._metadata(engine_config),
        )

        if not self._run_benchmark_request(request):
            return None
        return self._load_metrics(output_file)

    def _load_metrics(self, output_file: str) -> Optional[Dict]:
        if not os.path.exists(output_file):
            self.logger.error("Results file not found after benchmark")
            return None
        with open(output_file, "r") as f:
            metrics = self._get_metrics_from_results(json.load(f))
        if not metrics:
            self.logger.error("Failed to extract metrics from results.")
            return None
        return metrics

    def _metadata(self, engine_config: str) -> Dict[str, Any]:
        return {
            "autotune": "true",
            "engine_name": self.config["engine"]["name"],
            "docker_engine_args": engine_config,
        }

    def _meets_goodput_criteria(self, metrics: Dict) -> Tuple[List[Dict], bool]:
        thresholds = self.config["scenario"].get("goodput_criteria", {})
        results = []
        for threshold_name, threshold_value in thresholds.items():
            if threshold_name.startswith("max_"):
                metric_name = threshold_name[4:]
                if metric_name not in metrics:
                    raise ValueError(f"Unknown goodput criterion: {threshold_name}")
                metric_value = metrics[metric_name]
                meets = metric_value <= threshold_value
            elif threshold_name.startswith("min_"):
                metric_name = threshold_name[4:]
                if metric_name not in metrics:
                    raise ValueError(f"Unknown goodput criterion: {threshold_name}")
                metric_value = metrics[metric_name]
                meets = metric_value >= threshold_value
            else:
                raise ValueError(f"Unknown goodput criterion: {threshold_name}")
            results.append({threshold_name: threshold_value, metric_name: metric_value, "meets": meets})
        return results, all(r["meets"] for r in results)

    def _find_optimal_rate(
        self, max_throughput: float, run_id: str, output_folder: str, engine_config: str
    ) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        rate = max_throughput * 0.90
        attempts = 0
        scenario = self.config["scenario"]
        while rate > 0.1 and attempts < scenario["max_rate_finding_attempts"]:
            metrics = self._run_rate_benchmark(rate, run_id, output_folder, engine_config)
            if not metrics:
                rate *= 1 - scenario["rate_decrease_factor"]
                attempts += 1
                continue

            self.logger.info(f"Throughput: {metrics['throughput']:.2f} req/s")
            goodput_checks, meets = self._meets_goodput_criteria(metrics)
            self.logger.info("Goodput SLOs checks:")
            self.logger.info(json.dumps(goodput_checks, indent=2))
            if meets:
                self.logger.info(f"Found optimal rate: {rate:.2f} req/s")
                return metrics, goodput_checks
            rate *= 1 - scenario["rate_decrease_factor"]
            attempts += 1
        return None, None

    def _generate_parameter_combinations(self) -> List[Dict]:
        engine_config = self.config["engine"]
        value_args_pool = engine_config.get("value_args_pool", {})
        action_args_pool = engine_config.get("action_args_pool", {})
        if not value_args_pool and not action_args_pool:
            self.logger.warning("No tunable parameters defined in config, only base args will be used.")
            return [{"value_args": {}, "action_args": {}}]

        value_param_names = list(value_args_pool.keys())
        action_param_names = list(action_args_pool.keys())
        value_combinations = list(product(*[value_args_pool[name] for name in value_param_names])) or [()]
        action_combinations = list(product(*[action_args_pool[name] for name in action_param_names])) or [()]
        combinations = []
        for value_combo in value_combinations:
            for action_combo in action_combinations:
                combination = {
                    "value_args": dict(zip(value_param_names, value_combo)),
                    "action_args": dict(zip(action_param_names, action_combo)),
                }
                if "tp-dp-combinations" in combination["value_args"]:
                    tp_dp_comb = combination["value_args"].pop("tp-dp-combinations")
                    combination["value_args"]["tensor_parallel_size"] = tp_dp_comb["tp"]
                    combination["value_args"]["data_parallel_size"] = tp_dp_comb["dp"]
                combinations.append(combination)
        self.logger.info(
            f"Generated {len(combinations)} parameter combinations to test for engine {engine_config['name']}."
        )
        return combinations

    def _failed_result(self, run_id: str, param_config: Dict, engine_args: List[str], reason: str) -> Dict:
        return {
            "run_id": run_id,
            "tunable_parameters_config": param_config,
            "engine_container_command": engine_args,
            "failure_reason": reason,
        }

    def run_auto_tune(self) -> Dict:
        self.logger.info("Starting auto-tune process...")
        engine_name = self.config["engine"]["name"]
        autotune_id = uuid.uuid4().hex[:4]
        engine_path = self.results_dir.joinpath(engine_name, f"run_{self.timestamp}_{autotune_id}")
        engine_path.mkdir(parents=True, exist_ok=True)
        param_combinations = self._generate_parameter_combinations()
        shutil.copy2(self.config_path, engine_path / "auto_tune_config.yaml")

        all_results = []
        failed_results = []
        for i, param_config in enumerate(param_combinations, 1):
            container = None
            engine_args = []
            run_id = f"{self.config['model'].replace('/', '--')}_{uuid.uuid4().hex[:4]}"
            self.logger.info(f"{'=' * 60}")
            self.logger.info(f"[{i}/{len(param_combinations)}] Testing parameter combination: {param_config}")

            try:
                engine_args = self._build_engine_args(param_config)
                container = self.engine_runner.launch(self._engine_spec(engine_args), name_prefix="autotune")
                if not container:
                    failed_results.append(self._failed_result(run_id, param_config, engine_args, "launch_failed"))
                    continue
                if not self.engine_runner.wait_ready(container, self.config["port"], timeout=700):
                    failed_results.append(self._failed_result(run_id, param_config, engine_args, "server_not_ready"))
                    continue

                engine_command = " ".join(str(arg) for arg in engine_args)
                metrics = self._run_throughput_benchmark(run_id, engine_path.as_posix(), engine_command)
                if not metrics:
                    failed_results.append(
                        self._failed_result(run_id, param_config, engine_args, "throughput_benchmark_failed")
                    )
                    continue

                self.logger.info(f"Throughput: {metrics['throughput']:.2f} req/s")
                goodput_checks, meets = self._meets_goodput_criteria(metrics)
                self.logger.info("Goodput SLOs checks:")
                self.logger.info(json.dumps(goodput_checks, indent=2))
                if not meets:
                    if (metrics["throughput"] * 0.90) <= self.best_throughput["throughput"]:
                        failed_results.append(
                            self._failed_result(run_id, param_config, engine_args, "goodput_not_met")
                        )
                        continue
                    metrics, goodput_checks = self._find_optimal_rate(
                        metrics["throughput"], run_id, engine_path.as_posix(), engine_command
                    )
                    if not metrics:
                        failed_results.append(
                            self._failed_result(run_id, param_config, engine_args, "goodput_not_met")
                        )
                        continue

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
                    self.best_throughput = {"throughput": metrics["throughput"], "run_index": len(all_results)}
                    result["is_best"] = True
                    if all_results and last_best is not None:
                        all_results[last_best]["is_best"] = False
                    self.logger.info(f"NEW BEST CONFIG! Throughput: {metrics['throughput']:.2f} req/s")
                all_results.append(result)
            except Exception as exc:
                self.logger.exception(f"Error testing parameter config {param_config}: {exc}")
                failed_results.append(self._failed_result(run_id, param_config, engine_args, str(exc)))
            finally:
                if container:
                    self.engine_runner.cleanup(container)

        summary = {
            "timestamp": self.timestamp,
            "config_file": self.config_path,
            "engine_name": engine_name,
            "benchmark_backend": self.benchmark_executor.name,
            "instance_info": self.config.get("instance_info", {}),
            "goodput_criteria": self.config["scenario"].get("goodput_criteria", {}),
            "scenario": self.config["scenario"],
            "all_results": all_results,
            "failed_results": failed_results,
        }
        results_file = engine_path / "auto_tune_results.json"
        with open(results_file, "w") as f:
            json.dump(summary, f, indent=2)

        if self.dataset_id:
            self.logger.info(f"Uploading results to Huggingface dataset {self.dataset_id}...\n")
            hf_api.upload_folder(
                folder_path=self.results_dir,
                path_in_repo=str(self.results_dir.relative_to(self.root_dir)),
                repo_id=self.dataset_id,
                token=self.hf_token,
                repo_type="dataset",
            )

        self.logger.info(f"{'=' * 60}")
        self.logger.info("AUTO-TUNE COMPLETE")
        self.logger.info(f"Tested {len(param_combinations)} parameter combinations.")
        self.logger.info(f"Succeeded: {len(all_results)}, Failed: {len(failed_results)}")
        if self.best_throughput["run_index"] is not None:
            best = all_results[self.best_throughput["run_index"]]
            self.logger.info(f"Best throughput: {best['metrics']['throughput']:.2f} req/s")
            self.logger.info(f"Best config: {best['tunable_parameters_config']}")
        self.logger.info(f"Results saved to: {results_file}")
        return summary
