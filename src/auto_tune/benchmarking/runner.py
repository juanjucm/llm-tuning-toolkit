"""Execute reusable GuideLLM suites against caller-managed deployments."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
import uuid
from collections.abc import Iterable
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from auto_tune.guidellm import GuideLLMError, GuideLLMRunner
from auto_tune.slos import evaluate_slos

from .config import BenchmarkCase, ModelProfile, load_suite, select_benchmarks


class BenchmarkSuite:
    """Run a benchmark-only suite against an endpoint managed by the caller."""

    def __init__(
        self,
        config_path: str,
        *,
        recipe: str,
        target: str,
        model: str,
        tokenizer_model: str | None = None,
        result_dir: str = "out",
        context_window: int | None = None,
        capabilities: Iterable[str] = (),
        benchmark_names: Iterable[str] = (),
        include_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        guidellm_command: str = "guidellm",
        show_console: bool = True,
    ) -> None:
        if not recipe.strip():
            raise ValueError("recipe may not be empty")
        parsed_target = urlparse(target)
        if parsed_target.scheme not in {"http", "https"} or not parsed_target.netloc:
            raise ValueError("target must be an absolute HTTP(S) URL")
        self.config_path = config_path
        self.suite = load_suite(config_path)
        self.recipe = recipe
        self.target = target.rstrip("/")
        self.hf_token = os.getenv("HF_TOKEN") or None
        self.model_profile = ModelProfile(
            model=model,
            tokenizer_model=tokenizer_model or model,
            context_window=context_window,
            capabilities=frozenset(capabilities),
        )
        self.root_dir = Path(result_dir)
        self.benchmark_names = _normalized_set(benchmark_names, lowercase=False)
        self.include_tags = _normalized_set(include_tags)
        self.exclude_tags = _normalized_set(exclude_tags)
        self.show_console = show_console
        self.guidellm = GuideLLMRunner(guidellm_command)
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.logger = logging.getLogger(f"{__name__}.BenchmarkSuite.{id(self)}")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False

    def run(self) -> dict[str, Any]:
        """Run selected cases, preserve skips and failures, and write one report."""
        selections = select_benchmarks(
            self.suite,
            self.model_profile,
            benchmark_names=self.benchmark_names,
            include_tags=self.include_tags,
            exclude_tags=self.exclude_tags,
        )
        if not any(reason is None for _, reason in selections):
            reasons = "; ".join(f"{case.name}: {reason}" for case, reason in selections)
            raise ValueError(f"No benchmarks selected for this model ({reasons})")

        run_dir = self.root_dir.joinpath(
            _path_component(self.suite.name),
            _path_component(self.recipe),
            f"run_{self.timestamp}_{uuid.uuid4().hex[:4]}",
        )
        raw_dir = run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.config_path, run_dir / "benchmark_suite.yaml")

        summary: dict[str, Any] = {
            "timestamp": self.timestamp,
            "suite": self.suite.name,
            "suite_description": self.suite.description,
            "recipe": self.recipe,
            "target": self.target,
            "model": self.model_profile.model_dump(mode="json"),
            "authentication": {
                "kind": "huggingface_token",
                "enabled": self.hf_token is not None,
            },
            "selection": {
                "benchmark_names": sorted(self.benchmark_names),
                "include_tags": sorted(self.include_tags),
                "exclude_tags": sorted(self.exclude_tags),
            },
            "artifacts": {
                "suite_config": "benchmark_suite.yaml",
                "raw_reports": "raw/",
            },
            "benchmarks": [],
        }
        for benchmark, skip_reason in selections:
            if skip_reason is not None:
                summary["benchmarks"].append(self._skipped_result(benchmark, skip_reason))
            else:
                summary["benchmarks"].append(self._run_benchmark(benchmark, raw_dir))

        report_path = run_dir / "benchmark_results.json"
        with report_path.open("w") as file:
            json.dump(summary, file, indent=2)
        summary["report_path"] = str(report_path)
        self._log_summary(summary)
        return summary

    def _run_benchmark(self, benchmark: BenchmarkCase, raw_dir: Path) -> dict[str, Any]:
        self.logger.info("Running benchmark: %s", benchmark.name)
        result = self._result_metadata(benchmark)
        result["raw_report"] = f"raw/{benchmark.name}.json"
        try:
            points = self.guidellm.run_points(
                target=self.target,
                backend=self._backend(benchmark),
                data=benchmark.data,
                profile=benchmark.profile,
                constraints=benchmark.constraints,
                output_path=raw_dir / f"{benchmark.name}.json",
                options=self._guidellm_options(benchmark),
                show_console=self.show_console,
            )
        except GuideLLMError as error:
            result.update(status="failed", error=str(error), load_points=[])
            self.logger.error("Benchmark %s failed: %s", benchmark.name, error)
            return result

        load_points = []
        for point in points:
            metrics = point["metrics"]
            if benchmark.slos:
                checks, slo_met = evaluate_slos(benchmark.slos, metrics)
            else:
                checks, slo_met = [], None
            load_points.append(
                {
                    "load": point["load"],
                    "metrics": metrics,
                    "slo_checks": checks,
                    "slo_met": slo_met,
                }
            )
        result.update(status="completed", load_points=load_points)
        return result

    def _backend(self, benchmark: BenchmarkCase) -> dict[str, Any]:
        backend = {
            "kind": "openai_http",
            **self.suite.backend,
            **benchmark.backend,
            "target": self.target,
            "model": self.model_profile.model,
        }
        if self.hf_token is not None:
            backend["api_key"] = self.hf_token
        return backend

    def _guidellm_options(self, benchmark: BenchmarkCase) -> dict[str, Any]:
        options = deepcopy(self.suite.guidellm_options)
        benchmark_options = deepcopy(benchmark.guidellm_options)
        suite_arguments = options.pop("arguments", {})
        benchmark_arguments = benchmark_options.pop("arguments", {})
        options.update(benchmark_options)
        arguments = {**suite_arguments, **benchmark_arguments}
        tokenizer = arguments.get("tokenizer")
        if tokenizer is None:
            arguments["tokenizer"] = {
                "kind": "huggingface_auto",
                "model": self.model_profile.tokenizer_model,
            }
        elif isinstance(tokenizer, dict) and tokenizer.get("kind") == "huggingface_auto":
            arguments["tokenizer"] = {**tokenizer, "model": self.model_profile.tokenizer_model}
        options["arguments"] = arguments
        return options

    @staticmethod
    def _result_metadata(benchmark: BenchmarkCase) -> dict[str, Any]:
        return {
            "name": benchmark.name,
            "description": benchmark.description,
            "tags": sorted(benchmark.tags),
            "profile": benchmark.profile,
            "constraints": benchmark.constraints,
            "slos": benchmark.slos,
        }

    def _skipped_result(self, benchmark: BenchmarkCase, reason: str) -> dict[str, Any]:
        result = self._result_metadata(benchmark)
        result.update(status="skipped", reason=reason, load_points=[])
        self.logger.info("Skipping benchmark %s: %s", benchmark.name, reason)
        return result

    def _log_summary(self, summary: dict[str, Any]) -> None:
        self.logger.info("BENCHMARK SUITE COMPLETE")
        for benchmark in summary["benchmarks"]:
            if benchmark["status"] == "skipped":
                self.logger.info("%s: SKIPPED - %s", benchmark["name"], benchmark["reason"])
                continue
            if benchmark["status"] == "failed":
                self.logger.info("%s: FAILED - %s", benchmark["name"], benchmark["error"])
                continue
            for index, point in enumerate(benchmark["load_points"], 1):
                metrics = point["metrics"]
                verdict = (
                    "not evaluated" if point["slo_met"] is None else ("met" if point["slo_met"] else "missed")
                )
                self.logger.info(
                    "%s[%d]: %.2f req/s, TTFT p99 %.1f ms, E2E p99 %.1f ms, SLOs %s",
                    benchmark["name"],
                    index,
                    metrics["throughput"],
                    metrics["ttft_p99_ms"],
                    metrics["e2e_p99_ms"],
                    verdict,
                )
        self.logger.info("Results saved to: %s", summary["report_path"])


def _normalized_set(values: Iterable[str], *, lowercase: bool = True) -> frozenset[str]:
    stripped = (value.strip() for value in values if value.strip())
    return frozenset(value.lower() if lowercase else value for value in stripped)


def _path_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "--", value).strip(".-_")
    return component[:120] or "unnamed"
