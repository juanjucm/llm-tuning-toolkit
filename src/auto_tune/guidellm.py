"""GuideLLM command construction and report normalization for auto-tune."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from guidellm.benchmark import GenerativeBenchmarksReport


class GuideLLMError(RuntimeError):
    """Raised when GuideLLM fails or does not produce a usable report."""


def _descriptor(value: dict[str, Any] | str) -> str:
    """Serialize a typed GuideLLM descriptor without losing nested settings."""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Normalize a GuideLLM v0.7 report into metrics consumed by SLO checks.

    The GuideLLM report schema is the source of truth here. In particular,
    request latency is recorded in seconds while TTFT and ITL are milliseconds.
    """
    try:
        parsed = GenerativeBenchmarksReport.model_validate(report)
    except Exception as error:
        raise GuideLLMError(f"Could not parse GuideLLM JSON report: {error}") from error
    if not parsed.benchmarks:
        raise GuideLLMError("GuideLLM report contains no benchmark entries")

    # Throughput profiles have a single benchmark; a sweep can have many.
    # Choose the strategy with the greatest successful request rate.
    result = max(
        parsed.benchmarks,
        key=lambda benchmark: benchmark.metrics.requests_per_second.successful.mean,
    )
    metrics = result.metrics
    totals = metrics.request_totals

    def percentile_value(percentiles: Any, percentile: int) -> float | None:
        """Read a percentile without assuming every GuideLLM version exposes it.

        GuideLLM's standard percentile set is version-dependent (for example,
        some releases provide p50/p75/p90/p95/p99 but not p60).  Reports may
        also represent percentile values as a mapping rather than attributes.
        """
        name = f"p{percentile}"
        value = getattr(percentiles, name, None)
        if value is not None:
            return float(value)

        if hasattr(percentiles, "model_dump"):
            percentiles = percentiles.model_dump()
        if isinstance(percentiles, dict):
            for key in (name, str(percentile), str(percentile / 100), percentile, percentile / 100):
                value = percentiles.get(key)
                if value is not None:
                    return float(value)
        return None

    def milliseconds(summary: Any, *, already_ms: bool) -> dict[str, float]:
        factor = 1 if already_ms else 1000
        values = {
            "avg_ms": summary.successful.mean * factor,
        }
        for percentile in (50, 60, 70, 80, 90, 95, 99):
            value = percentile_value(summary.successful.percentiles, percentile)
            if value is not None:
                values[f"p{percentile}_ms"] = value * factor
        return values

    normalized: dict[str, float] = {
        "throughput": metrics.requests_per_second.successful.mean,
        "total_requests": float(totals.total),
        "successful_requests": float(totals.successful),
        "failed_requests": float(totals.errored + totals.incomplete),
        "success_rate": float(totals.successful / totals.total) if totals.total else 0.0,
        "output_tokens_per_second": metrics.output_tokens_per_second.successful.mean,
        "total_tokens_per_second": metrics.tokens_per_second.successful.mean,
    }
    for name, summary, already_ms in (
        ("ttft", metrics.time_to_first_token_ms, True),
        ("e2e", metrics.request_latency, False),
        ("itl", metrics.inter_token_latency_ms, True),
    ):
        normalized.update({f"{name}_{key}": value for key, value in milliseconds(summary, already_ms=already_ms).items()})
    return normalized


class GuideLLMRunner:
    """Run GuideLLM against a local OpenAI-compatible engine."""

    def __init__(self, command: str = "guidellm") -> None:
        self.command = command

    def run(
        self,
        *,
        target: str,
        data: list[dict[str, Any] | str],
        profile: dict[str, Any],
        constraints: list[dict[str, Any] | str],
        output_path: Path,
        backend: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        backend_config: dict[str, Any] | str = backend or {"kind": "openai_http", "target": target}
        if isinstance(backend_config, dict):
            backend_config = {**backend_config, "target": backend_config.get("target", target)}
        cmd = [
            self.command,
            "run",
            "--backend",
            _descriptor(backend_config),
            "--profile",
            _descriptor(profile),
            "--output",
            _descriptor({"kind": "json", "path": str(output_path)}),
            "--metrics",
            _descriptor({"kind": "generative", "sample_size": (options or {}).get("sample_size", 0)}),
        ]
        for constraint in constraints:
            cmd.extend(["--constraint", _descriptor(constraint)])
        for item in data:
            cmd.extend(["--data", _descriptor(item)])
        for option, value in (options or {}).get("arguments", {}).items():
            flag = f"--{option.replace('_', '-')}"
            if isinstance(value, list):
                for item in value:
                    cmd.extend([flag, _descriptor(item) if isinstance(item, dict) else str(item)])
            elif isinstance(value, bool):
                if value:
                    cmd.append(flag)
            elif value is not None:
                cmd.extend([flag, _descriptor(value) if isinstance(value, dict) else str(value)])

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as error:
            raise GuideLLMError("GuideLLM is not installed; install the project dependencies first") from error
        except subprocess.CalledProcessError as error:
            raise GuideLLMError(f"GuideLLM failed with exit code {error.returncode}") from error
        if not output_path.exists():
            raise GuideLLMError(f"GuideLLM completed without creating {output_path}")
        with output_path.open() as file:
            return extract_metrics(json.load(file))
