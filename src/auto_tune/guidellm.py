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
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from guidellm.benchmark import GenerativeBenchmarksReport


class GuideLLMError(RuntimeError):
    """Raised when GuideLLM fails or does not produce a usable report."""


class _TerminalScreen:
    """Minimal VT100 screen used to recover Rich Live output from a PTY."""

    def __init__(self, rows: int, columns: int) -> None:
        self.rows = rows
        self.columns = columns
        self._lines = [[" "] * columns for _ in range(rows)]
        self._row = 0
        self._column = 0
        self._saved_cursor = (0, 0)
        self._pending = ""

    def feed(self, chunk: str) -> None:
        text = self._pending + chunk
        self._pending = ""
        index = 0
        while index < len(text):
            character = text[index]
            if character == "\x1b":
                if index + 1 >= len(text):
                    self._pending = text[index:]
                    break
                if text[index + 1] == "[":
                    end = index + 2
                    while end < len(text) and not ("@" <= text[end] <= "~"):
                        end += 1
                    if end >= len(text):
                        self._pending = text[index:]
                        break
                    self._apply_csi(text[index + 2 : end], text[end])
                    index = end + 1
                    continue
                index += 2
                continue
            if character == "\r":
                self._column = 0
            elif character == "\n":
                self._line_feed()
            elif character == "\b":
                self._column = max(0, self._column - 1)
            elif character >= " ":
                self._write(character)
            index += 1

    def benchmark_progress(self) -> str | None:
        """Return only GuideLLM's interactive benchmark block."""
        lines = ["".join(line).rstrip() for line in self._lines]
        starts = [index for index, line in enumerate(lines) if "Benchmarks" in line]
        if not starts:
            return None
        start = starts[-1]
        for end in range(start, len(lines)):
            if "Generating..." in lines[end]:
                return "\n".join(lines[start : end + 1]).strip()
        return None

    def _write(self, character: str) -> None:
        if unicodedata.combining(character):
            if self._column:
                self._lines[self._row][self._column - 1] += character
            return
        width = 2 if unicodedata.east_asian_width(character) in {"F", "W"} else 1
        if self._column >= self.columns:
            self._column = 0
            self._line_feed()
        self._lines[self._row][self._column] = character
        if width == 2 and self._column + 1 < self.columns:
            self._lines[self._row][self._column + 1] = ""
        self._column += width

    def _line_feed(self) -> None:
        if self._row == self.rows - 1:
            self._lines.pop(0)
            self._lines.append([" "] * self.columns)
        else:
            self._row += 1

    def _apply_csi(self, parameters: str, command: str) -> None:
        clean_parameters = parameters.lstrip("?")
        values = [int(value) if value else 0 for value in clean_parameters.split(";")] if clean_parameters else []
        amount = values[0] if values and values[0] else 1
        if command == "A":
            self._row = max(0, self._row - amount)
        elif command == "B":
            self._row = min(self.rows - 1, self._row + amount)
        elif command == "C":
            self._column = min(self.columns - 1, self._column + amount)
        elif command == "D":
            self._column = max(0, self._column - amount)
        elif command == "E":
            self._row = min(self.rows - 1, self._row + amount)
            self._column = 0
        elif command == "F":
            self._row = max(0, self._row - amount)
            self._column = 0
        elif command == "G":
            self._column = min(self.columns - 1, max(0, amount - 1))
        elif command in {"H", "f"}:
            row = values[0] if values and values[0] else 1
            column = values[1] if len(values) > 1 and values[1] else 1
            self._row = min(self.rows - 1, max(0, row - 1))
            self._column = min(self.columns - 1, max(0, column - 1))
        elif command == "J":
            self._erase_display(values[0] if values else 0)
        elif command == "K":
            self._erase_line(values[0] if values else 0)
        elif command == "s":
            self._saved_cursor = (self._row, self._column)
        elif command == "u":
            self._row, self._column = self._saved_cursor

    def _erase_line(self, mode: int) -> None:
        if mode == 1:
            start, end = 0, self._column + 1
        elif mode == 2:
            start, end = 0, self.columns
        else:
            start, end = self._column, self.columns
        self._lines[self._row][start:end] = [" "] * (end - start)

    def _erase_display(self, mode: int) -> None:
        if mode in {2, 3}:
            self._lines = [[" "] * self.columns for _ in range(self.rows)]
            return
        if mode == 1:
            for row in range(self._row):
                self._lines[row] = [" "] * self.columns
            self._lines[self._row][: self._column + 1] = [" "] * (self._column + 1)
            return
        self._lines[self._row][self._column :] = [" "] * (self.columns - self._column)
        for row in range(self._row + 1, self.rows):
            self._lines[row] = [" "] * self.columns


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
        on_output: Callable[[str], None] | None = None,
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

        if on_output is None:
            try:
                subprocess.run(cmd, check=True)
            except FileNotFoundError as error:
                raise GuideLLMError("GuideLLM is not installed; install the project dependencies first") from error
            except subprocess.CalledProcessError as error:
                raise GuideLLMError(f"GuideLLM failed with exit code {error.returncode}") from error
            return self._read_metrics(output_path)

        returncode = self._run_with_live_output(cmd, on_output)
        if returncode:
            raise GuideLLMError(f"GuideLLM failed with exit code {returncode}")

        return self._read_metrics(output_path)

    @staticmethod
    def _read_metrics(output_path: Path) -> dict[str, float]:
        """Read and normalize the JSON report created by GuideLLM."""
        if not output_path.exists():
            raise GuideLLMError(f"GuideLLM completed without creating {output_path}")
        with output_path.open() as file:
            return extract_metrics(json.load(file))

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
        screen = _TerminalScreen(rows=rows, columns=columns)
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
                progress = screen.benchmark_progress()
                if progress and progress != last_progress:
                    on_output(progress)
                    last_progress = progress
        finally:
            os.close(master_fd)
        return process.wait()
