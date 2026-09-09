"""GuideLLM command construction and report normalization for auto-tune."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import shutil
import struct
import subprocess
import termios
from collections.abc import Callable
from pathlib import Path
from typing import Any

from guidellm.benchmark import GenerativeBenchmarksReport
from guidellm.schemas.statistics import Percentiles

from auto_tune.terminal import TerminalScreen


class GuideLLMError(RuntimeError):
    """Raised when GuideLLM fails or does not produce a usable report."""


# Emit exactly the percentiles GuideLLM reports, so an SLO name cannot refer to a
# percentile the report does not carry.
PERCENTILE_NAMES: tuple[str, ...] = tuple(Percentiles.model_fields)
SCALAR_METRIC_NAMES: tuple[str, ...] = (
    "throughput",
    "total_requests",
    "successful_requests",
    "failed_requests",
    "success_rate",
    "output_tokens_per_second",
    "total_tokens_per_second",
)
LATENCY_METRIC_PREFIXES: tuple[str, ...] = ("ttft", "e2e", "itl")


def metric_names() -> frozenset[str]:
    """Every key extract_metrics can produce, for config-time SLO validation."""
    return frozenset(
        SCALAR_METRIC_NAMES
        + tuple(
            f"{prefix}_{suffix}"
            for prefix in LATENCY_METRIC_PREFIXES
            for suffix in ("avg_ms", *(f"{name}_ms" for name in PERCENTILE_NAMES))
        )
    )


def _descriptor(value: dict[str, Any] | str) -> str:
    """Serialize a typed GuideLLM descriptor without losing nested settings."""
    return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))


def _milliseconds(summary: Any, *, already_ms: bool) -> dict[str, float]:
    """Read a latency summary's mean and every reported percentile, in milliseconds."""
    factor = 1 if already_ms else 1000
    values = {"avg_ms": summary.successful.mean * factor}
    for name in PERCENTILE_NAMES:
        values[f"{name}_ms"] = float(getattr(summary.successful.percentiles, name)) * factor
    return values


def _normalize(result: Any) -> dict[str, float]:
    """Normalize one GuideLLM benchmark into the metric names SLOs refer to."""
    metrics = result.metrics
    totals = metrics.request_totals
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
        for key, value in _milliseconds(summary, already_ms=already_ms).items():
            normalized[f"{name}_{key}"] = value
    return normalized


def _parse_report(report: dict[str, Any]) -> GenerativeBenchmarksReport:
    try:
        parsed = GenerativeBenchmarksReport.model_validate(report)
    except Exception as error:
        raise GuideLLMError(f"Could not parse GuideLLM JSON report: {error}") from error
    if not parsed.benchmarks:
        raise GuideLLMError("GuideLLM report contains no benchmark entries")
    return parsed


def _load_config(result: Any) -> dict[str, Any]:
    """Expose the concrete load attached to a GuideLLM benchmark result."""
    strategy = result.config.strategy.model_dump(exclude_none=True)
    return {"kind": strategy.pop("type_"), **strategy}


def extract_all_results(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized metrics with the concrete load for every benchmark point."""
    parsed = _parse_report(report)
    points = [{"load": _load_config(entry), "metrics": _normalize(entry)} for entry in parsed.benchmarks]
    return sorted(points, key=lambda point: point["metrics"]["throughput"])


def extract_all_metrics(report: dict[str, Any]) -> list[dict[str, float]]:
    """Normalize every benchmark in a GuideLLM v0.7 report, by ascending throughput.

    The GuideLLM report schema is the source of truth here. In particular,
    request latency is recorded in seconds while TTFT and ITL are milliseconds.
    """
    parsed = _parse_report(report)
    # A concurrent profile reports one benchmark per configured stream count. SLO
    # evaluation needs all of them: the most saturated point is the one most likely
    # to violate a latency SLO, so collapsing to it discards the configurations
    # that would have qualified at a lower load.
    return sorted((_normalize(entry) for entry in parsed.benchmarks), key=lambda metrics: metrics["throughput"])


def extract_metrics(report: dict[str, Any]) -> dict[str, float]:
    """Normalize the highest-throughput benchmark of a GuideLLM v0.7 report."""
    return extract_all_metrics(report)[-1]


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
        on_output: Callable[[str], None] | None = None,
        show_console: bool = True,
    ) -> dict[str, float]:
        """Run one benchmark and return its highest-throughput load point."""
        return extract_metrics(
            self._execute(
                target=target,
                data=data,
                profile=profile,
                constraints=constraints,
                output_path=output_path,
                backend=backend,
                options=options,
                on_output=on_output,
                show_console=show_console,
            )
        )

    def run_points(
        self,
        *,
        target: str,
        data: list[dict[str, Any] | str],
        profile: dict[str, Any],
        constraints: list[dict[str, Any] | str],
        output_path: Path,
        backend: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
        on_output: Callable[[str], None] | None = None,
        show_console: bool = True,
    ) -> list[dict[str, Any]]:
        """Run one benchmark and retain each point's concrete load parameters."""
        return extract_all_results(
            self._execute(
                target=target,
                data=data,
                profile=profile,
                constraints=constraints,
                output_path=output_path,
                backend=backend,
                options=options,
                on_output=on_output,
                show_console=show_console,
            )
        )

    def run_all(
        self,
        *,
        target: str,
        data: list[dict[str, Any] | str],
        profile: dict[str, Any],
        constraints: list[dict[str, Any] | str],
        output_path: Path,
        backend: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
        on_output: Callable[[str], None] | None = None,
        show_console: bool = True,
    ) -> list[dict[str, float]]:
        """Run one benchmark and return every load point it measured, ascending."""
        return extract_all_metrics(
            self._execute(
                target=target,
                data=data,
                profile=profile,
                constraints=constraints,
                output_path=output_path,
                backend=backend,
                options=options,
                on_output=on_output,
                show_console=show_console,
            )
        )

    def _execute(
        self,
        *,
        target: str,
        data: list[dict[str, Any] | str],
        profile: dict[str, Any],
        constraints: list[dict[str, Any] | str],
        output_path: Path,
        backend: dict[str, Any] | str | None = None,
        options: dict[str, Any] | None = None,
        on_output: Callable[[str], None] | None = None,
        show_console: bool = True,
    ) -> dict[str, Any]:
        """Build and run the GuideLLM command line, returning its JSON report."""
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

        if not show_console:
            cmd.extend(["--disable-console", "--disable-console-interactive"])

        if on_output is None:
            try:
                run_options: dict[str, Any] = {"check": True}
                if not show_console:
                    run_options.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                subprocess.run(cmd, **run_options)
            except FileNotFoundError as error:
                raise GuideLLMError("GuideLLM is not installed; install the project dependencies first") from error
            except subprocess.CalledProcessError as error:
                raise GuideLLMError(f"GuideLLM failed with exit code {error.returncode}") from error
            return self._read_report(output_path)

        returncode = self._run_with_live_output(cmd, on_output)
        if returncode:
            raise GuideLLMError(f"GuideLLM failed with exit code {returncode}")

        return self._read_report(output_path)

    @staticmethod
    def _read_report(output_path: Path) -> dict[str, Any]:
        """Read the JSON report created by GuideLLM."""
        if not output_path.exists():
            raise GuideLLMError(f"GuideLLM completed without creating {output_path}")
        with output_path.open() as file:
            return json.load(file)

    @staticmethod
    def _run_with_live_output(cmd: list[str], on_output: Callable[[str], None]) -> int:
        """Run GuideLLM in a PTY and publish its current Rich progress screen."""
        master_fd, slave_fd = pty.openpty()
        columns = max(60, shutil.get_terminal_size((120, 40)).columns - 4)
        rows = 60
        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, columns, 0, 0))
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=slave_fd,
                stderr=slave_fd,
                close_fds=True,
            )
        except FileNotFoundError as error:
            os.close(master_fd)
            os.close(slave_fd)
            raise GuideLLMError("GuideLLM is not installed; install the project dependencies first") from error

        os.close(slave_fd)
        screen = TerminalScreen(rows=rows, columns=columns)
        last_progress = None
        try:
            while True:
                try:
                    chunk = os.read(master_fd, 65536)
                except OSError as error:
                    # PTYs report EIO after the child closes the slave side.
                    if error.errno == errno.EIO:
                        break
                    raise GuideLLMError(f"Could not read GuideLLM terminal output: {error}") from error
                if not chunk:
                    break
                screen.feed(chunk.decode("utf-8", errors="replace"))
                progress = screen.extract_block("Benchmarks", "Generating...")
                if progress and progress != last_progress:
                    on_output(progress)
                    last_progress = progress
        finally:
            os.close(master_fd)
        return process.wait()
