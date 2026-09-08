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

from auto_tune.display_format import (
    format_config,
    format_constraints,
    format_data,
    format_latency_pair,
    format_metric,
    format_percentage,
    format_slos,
    format_workload,
)


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
        self._workload = format_workload(scenario.get("load", {}))
        self._stop_constraint = format_constraints(scenario.get("constraints", []))
        self._data_summary = format_data(scenario.get("data", []))
        self._target_slos = format_slos(scenario.get("slos", {}))
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
        self._current_config = f"{index}/{total}  {format_config(parameters)}"
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
        self._best_config = format_config(parameters)
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
            format_metric(metrics, "throughput", "req/s"),
            "Output throughput",
            format_metric(metrics, "output_tokens_per_second", "tok/s"),
            "Success rate",
            format_percentage(metrics.get("success_rate")),
        )
        details.add_row(
            "TTFT avg / p99",
            format_latency_pair(metrics, "ttft"),
            "ITL avg / p99",
            format_latency_pair(metrics, "itl"),
            "E2E avg / p99",
            format_latency_pair(metrics, "e2e"),
        )
        return Group(configuration, details)

    def _refresh(self) -> None:
        if self._started:
            self._live.update(self._render(), refresh=True)
