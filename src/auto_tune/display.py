"""Rich terminal display for an auto-tune run."""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text


class AutoTuneDisplay:
    """Keep auto-tune status and GuideLLM output in one stable terminal view."""

    def __init__(self) -> None:
        self.progress = Progress(
            TextColumn("[bold cyan]Configurations"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("[dim]{task.completed}/{task.total}"),
        )
        self._guidellm_progress = ""
        self._status = "Preparing auto-tune run"
        self._status_style = "white"
        self._current_config = "–"
        self._best_throughput: float | None = None
        self._started = False
        self._task_id: int | None = None
        self._live = Live(self._render(), refresh_per_second=8, transient=False)

    def start(self, *, total_configs: int, scenario_name: str, model: str) -> None:
        self._task_id = self.progress.add_task("Configurations", total=total_configs)
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
        value_args = parameters.get("value_args", {})
        action_args = [name for name, enabled in parameters.get("action_args", {}).items() if enabled]
        parameter_text = ", ".join(f"{name}={value}" for name, value in value_args.items())
        if action_args:
            parameter_text = ", ".join(filter(None, [parameter_text, *action_args]))
        self._current_config = f"{index}/{total}  {parameter_text or 'base arguments'}"
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

    def set_best(self, throughput: float) -> None:
        self._best_throughput = throughput
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
        summary = Table.grid(expand=True)
        summary.add_column(style="bold cyan", width=16)
        summary.add_column()
        summary.add_row("Status", Text(self._status, style=self._status_style))
        summary.add_row("Current", self._current_config)
        best = "–" if self._best_throughput is None else f"{self._best_throughput:.2f} req/s"
        summary.add_row("Best throughput", Text(best, style="bold green" if self._best_throughput is not None else "dim"))

        guidellm = (
            Text(self._guidellm_progress)
            if self._guidellm_progress
            else Text("Waiting for GuideLLM benchmark progress…", style="dim")
        )
        return Group(
            Panel(Group(summary, self.progress), title="[bold]Auto-tune[/bold]", border_style="cyan"),
            Panel(guidellm, title="[bold magenta]GuideLLM benchmark[/bold magenta]", border_style="magenta"),
        )

    def _refresh(self) -> None:
        if self._started:
            self._live.update(self._render(), refresh=True)
