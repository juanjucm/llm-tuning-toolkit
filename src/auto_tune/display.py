"""Rich terminal display for an auto-tune run."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text


class AutoTuneDisplay:
    """Keep auto-tune status and GuideLLM output in one stable terminal view."""

    def __init__(self) -> None:
        self.progress = Progress(
            TextColumn("[bold cyan]Configurations"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.completed}/{task.total}"),
        )
        self._guidellm_progress = ""
        self._status = "Preparing auto-tune run"
        self._status_style = "white"
        self._current_config = "–"
        self._model = "–"
        self._scenario_name = "–"
        self._workload = "–"
        self._stop_constraint = "–"
        self._data_summary = "–"
        self._target_slos: list[Text] = []
        self._best_metrics: dict[str, float] | None = None
        self._best_config = "–"
        self._started = False
        self._task_id: int | None = None
        self._live = Live(self._render(), refresh_per_second=8, transient=False)

    def start(
        self,
        *,
        total_configs: int,
        scenario_name: str,
        model: str,
        scenario: dict[str, Any] | None = None,
    ) -> None:
        self._task_id = self.progress.add_task("Configurations", total=total_configs)
        self._model = model
        self._scenario_name = scenario_name
        scenario = scenario or {}
        self._workload = self._format_workload(scenario.get("load", {}))
        self._stop_constraint = self._format_constraints(scenario.get("constraints", []))
        self._data_summary = self._format_data(scenario.get("data", []))
        self._target_slos = self._format_slos(scenario.get("slos", {}))
        self._status = "Starting first configuration"
        self._status_style = "cyan"
        self._started = True
        self._live.start()
        self._refresh()

    def stop(self) -> None:
        if self._started:
            self._live.stop()
            self._started = False

    def begin_config(self, index: int, total: int, parameters: dict[str, Any]) -> None:
        self._current_config = f"{index}/{total}  {self._format_config(parameters)}"
        self._status = "Launching engine"
        self._status_style = "cyan"
        self._guidellm_progress = ""
        self._refresh()

    def complete_config(self, index: int) -> None:
        if self._task_id is not None:
            self.progress.update(self._task_id, completed=index)
        self._status = f"Configuration {index} complete"
        self._status_style = "green"
        self._refresh()

    def set_status(self, status: str) -> None:
        self._status = status
        self._status_style = "cyan"
        self._refresh()

    def set_best(self, metrics: dict[str, float], parameters: dict[str, Any]) -> None:
        self._best_metrics = dict(metrics)
        self._best_config = self._format_config(parameters)
        self._refresh()

    def log(self, message: str, level: str = "INFO") -> None:
        if level in {"ERROR", "WARNING"}:
            self._status = message.splitlines()[0]
            self._status_style = "bold red" if level == "ERROR" else "yellow"
            self._refresh()

    def set_guidellm_progress(self, progress: str) -> None:
        if progress:
            self._guidellm_progress = progress.rstrip()
            self._refresh()

    def _render(self) -> Group:
        identity = Table.grid(expand=True)
        identity.add_column(style="bold cyan", width=16)
        identity.add_column(ratio=1)
        identity.add_row("Model", self._model)
        identity.add_row("Scenario", self._scenario_name)
        identity.add_row("", "")
        identity.add_row("Workload", self._workload)
        identity.add_row("Stop constraint", self._stop_constraint)
        identity.add_row("Data", self._data_summary)

        slos = Table.grid(expand=True, padding=(0, 3))
        slos.add_column(ratio=1)
        slos.add_column(ratio=1)
        if self._target_slos:
            for index in range(0, len(self._target_slos), 2):
                second = self._target_slos[index + 1] if index + 1 < len(self._target_slos) else Text("")
                slos.add_row(self._target_slos[index], second)
        else:
            slos.add_row(Text("No target SLOs configured.", style="dim"), Text(""))

        activity = Table.grid(expand=True)
        activity.add_column(style="bold cyan", width=16)
        activity.add_column(ratio=1)
        activity.add_row("Current", self._current_config)
        activity.add_row("Status", Text(self._status, style=self._status_style))

        best_run = self._render_best_run()

        guidellm = (
            Text(self._guidellm_progress)
            if self._guidellm_progress
            else Text("Waiting for GuideLLM benchmark progress…", style="dim")
        )
        return Group(
            Panel(
                Group(
                    identity,
                    Text("Target SLOs", style="bold cyan"),
                    slos,
                    Rule(style="dim cyan"),
                    self.progress,
                    activity,
                    Text(""),
                    Rule("Best run", align="left", style="green"),
                    best_run,
                ),
                title="[bold]Auto-tune[/bold]",
                border_style="cyan",
            ),
            Panel(guidellm, title="[bold magenta]Live benchmark[/bold magenta]", border_style="magenta"),
        )

    def _render_best_run(self) -> Group | Text:
        if self._best_metrics is None:
            return Text("No SLO-compliant run yet.", style="dim")

        metrics = self._best_metrics
        configuration = Table.grid(expand=True)
        configuration.add_column(style="bold cyan", width=16)
        configuration.add_column(ratio=1)
        configuration.add_row("Configuration", Text(self._best_config, style="green"))

        details = Table.grid(expand=True)
        details.add_column(style="bold cyan", width=16)
        details.add_column(ratio=1)
        details.add_column(style="bold cyan", width=18)
        details.add_column(ratio=1)
        details.add_column(style="bold cyan", width=18)
        details.add_column(ratio=1)
        details.add_row(
            "Throughput",
            self._format_metric(metrics, "throughput", "req/s"),
            "Output throughput",
            self._format_metric(metrics, "output_tokens_per_second", "tok/s"),
            "Success rate",
            self._format_percentage(metrics.get("success_rate")),
        )
        details.add_row(
            "TTFT avg / p99",
            self._format_latency_pair(metrics, "ttft"),
            "ITL avg / p99",
            self._format_latency_pair(metrics, "itl"),
            "E2E avg / p99",
            self._format_latency_pair(metrics, "e2e"),
        )
        return Group(configuration, details)

    @staticmethod
    def _format_config(parameters: dict[str, Any]) -> str:
        value_args = parameters.get("value_args", {})
        action_args = [name for name, enabled in parameters.get("action_args", {}).items() if enabled]
        values = [f"{name}={value}" for name, value in value_args.items()]
        return ", ".join([*values, *action_args]) or "base arguments"

    @staticmethod
    def _format_workload(load: dict[str, Any]) -> str:
        kind = str(load.get("kind", "throughput"))
        details: list[str] = []
        labels = {
            "throughput": "Throughput",
            "concurrent": "Concurrent",
            "constant": "Constant rate",
            "poisson": "Poisson arrivals",
            "replay": "Trace replay",
        }
        if kind == "throughput" and load.get("max_concurrency") is not None:
            details.append(f"max concurrency {load['max_concurrency']}")
        elif kind == "concurrent":
            if load.get("streams") is not None:
                details.append(f"{load['streams']} streams")
            if load.get("turns") is not None:
                details.append(f"{load['turns']} turns")
            if load.get("delay") is not None:
                details.append(f"{load['delay']} s think time")
        elif kind in {"constant", "poisson"}:
            if load.get("rate") is not None:
                details.append(f"{load['rate']} req/s")
            if load.get("max_concurrency") is not None:
                details.append(f"max concurrency {load['max_concurrency']}")
        elif kind == "replay" and load.get("time_scale") is not None:
            details.append(f"{load['time_scale']}× speed")
        else:
            details.extend(f"{name}={value}" for name, value in load.items() if name != "kind")
        return " · ".join(filter(None, [labels.get(kind, kind.replace("_", " ").title()), *details]))

    @classmethod
    def _format_constraints(cls, constraints: object) -> str:
        if isinstance(constraints, (dict, str)):
            constraints = [constraints]
        if not isinstance(constraints, list) or not constraints:
            return "–"
        return " · ".join(cls._format_constraint(item) for item in constraints)

    @staticmethod
    def _format_constraint(constraint: object) -> str:
        if isinstance(constraint, str):
            return constraint
        if not isinstance(constraint, dict):
            return str(constraint)
        kind = str(constraint.get("kind", "constraint"))
        if kind == "max_requests":
            count = constraint.get("count")
            return f"Max requests: {count:,}" if isinstance(count, (int, float)) else "Max requests"
        if kind == "max_duration":
            seconds = constraint.get("seconds")
            return f"Max duration: {seconds:g} s" if isinstance(seconds, (int, float)) else "Max duration"
        if kind == "max_errors":
            count = constraint.get("count")
            return f"Max errors: {count:,}" if isinstance(count, (int, float)) else "Max errors"
        if kind in {"max_error_rate", "max_global_error_rate"}:
            rate = constraint.get("rate", constraint.get("value"))
            label = "Max error rate" if kind == "max_error_rate" else "Max global error rate"
            return f"{label}: {rate * 100:.1f}%" if isinstance(rate, (int, float)) else label
        labels = {
            "over_saturation": "Over-saturation guard",
        }
        values = [str(value) for name, value in constraint.items() if name != "kind"]
        return ": ".join(filter(None, [labels.get(kind, kind.replace("_", " ").title()), ", ".join(values)]))

    @staticmethod
    def _format_data(data: object) -> str:
        if not isinstance(data, list) or not data:
            return "–"
        first = data[0]
        if isinstance(first, str):
            summary = first
        elif isinstance(first, dict):
            kind = str(first.get("kind", "data"))
            if kind == "synthetic_text":
                prompt = first.get("prompt_tokens")
                output = first.get("output_tokens")
                parts = ["Synthetic text"]
                if prompt is not None:
                    parts.append(f"input {prompt} tok")
                if output is not None:
                    parts.append(f"output {output} tok")
                summary = " · ".join(parts)
            elif kind == "synthetic_image":
                dimensions = "×".join(str(first[key]) for key in ("width", "height") if first.get(key) is not None)
                summary = " · ".join(filter(None, ["Synthetic image", dimensions]))
            elif kind == "json_file":
                summary = f"JSON file · {first.get('path', 'path not set')}"
            else:
                summary = kind.replace("_", " ").title()
        else:
            summary = str(first)
        if len(data) > 1:
            summary += f" · +{len(data) - 1} source{'s' if len(data) > 2 else ''}"
        return summary

    @classmethod
    def _format_slos(cls, slos: object) -> list[Text]:
        if not isinstance(slos, dict):
            return []
        return [cls._format_slo(name, value) for name, value in slos.items()]

    @staticmethod
    def _format_slo(name: str, value: object) -> Text:
        if name.startswith("min_"):
            operator, metric = "≥", name.removeprefix("min_")
        elif name.startswith("max_"):
            operator, metric = "≤", name.removeprefix("max_")
        else:
            operator, metric = "=", name

        labels = {
            "success_rate": "Success rate",
            "throughput": "Request throughput",
            "output_tokens_per_second": "Output throughput",
            "total_tokens_per_second": "Total-token throughput",
        }
        label = labels.get(metric)
        if label is None:
            parts = metric.split("_")
            prefixes = {"ttft": "TTFT", "itl": "ITL", "e2e": "E2E"}
            if parts[0] in prefixes:
                label = " ".join([prefixes[parts[0]], *[part for part in parts[1:] if part != "ms"]])
            else:
                label = metric.replace("_", " ").title()

        if isinstance(value, (int, float)):
            if metric == "success_rate":
                value_text = f"{value * 100:.1f}%"
            elif metric.endswith("_ms"):
                value_text = f"{value:g} ms"
            elif metric in {"throughput"}:
                value_text = f"{value:g} req/s"
            elif metric in {"output_tokens_per_second", "total_tokens_per_second"}:
                value_text = f"{value:g} tok/s"
            else:
                value_text = f"{value:g}"
        else:
            value_text = str(value)
        return Text.assemble((f"• {label}", "cyan"), f"  {operator} {value_text}")

    @staticmethod
    def _format_metric(metrics: dict[str, float], name: str, unit: str) -> str:
        value = metrics.get(name)
        return "–" if value is None else f"{value:.2f} {unit}"

    @staticmethod
    def _format_percentage(value: float | None) -> str:
        return "–" if value is None else f"{value * 100:.1f}%"

    @staticmethod
    def _format_latency_pair(metrics: dict[str, float], prefix: str) -> str:
        average = metrics.get(f"{prefix}_avg_ms")
        p99 = metrics.get(f"{prefix}_p99_ms")
        if average is None and p99 is None:
            return "–"
        average_text = "–" if average is None else f"{average:.1f}"
        p99_text = "–" if p99 is None else f"{p99:.1f}"
        return f"{average_text} / {p99_text} ms"

    def _refresh(self) -> None:
        if self._started:
            self._live.update(self._render(), refresh=True)
