"""Rich terminal display for an auto-tune run."""

from __future__ import annotations

from collections import deque
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
        self._events: deque[Text] = deque(maxlen=7)
        self._guidellm_output: deque[Text] = deque(maxlen=12)
        self._status = "Preparing auto-tune run"
        self._current_config = "–"
        self._best_throughput: float | None = None
        self._started = False
        self._task_id: int | None = None
        self._live = Live(self._render(), refresh_per_second=8, transient=False)

    def start(self, *, total_configs: int, scenario_name: str, model: str) -> None:
        self._task_id = self.progress.add_task("Configurations", total=total_configs)
        self._status = "Starting first configuration"
        self._events.append(Text(f"Scenario: {scenario_name}  •  Model: {model}", style="dim"))
        self._started = True
        self._live.start()
        self._refresh()

    def stop(self) -> None:
        if self._started:
            self._live.stop()
            self._started = False

    def begin_config(self, index: int, total: int, parameters: dict[str, Any]) -> None:
        self._current_config = f"{index}/{total}  {parameters}"
        self._status = "Launching engine"
        self._refresh()

    def complete_config(self, index: int) -> None:
        if self._task_id is not None:
            self.progress.update(self._task_id, completed=index)
        self._status = "Preparing next configuration"
        self._refresh()

    def set_status(self, status: str) -> None:
        self._status = status
        self._refresh()

    def set_best(self, throughput: float) -> None:
        self._best_throughput = throughput
        self._refresh()

    def log(self, message: str, level: str = "INFO") -> None:
        styles = {"ERROR": "bold red", "WARNING": "yellow", "INFO": "white"}
        for line in message.splitlines() or [""]:
            self._events.append(Text(line, style=styles.get(level, "white")))
        self._refresh()

    def add_guidellm_output(self, line: str) -> None:
        line = line.rstrip()
        if line:
            self._guidellm_output.append(Text.from_ansi(line))
            self._refresh()

    def _render(self) -> Group:
        summary = Table.grid(expand=True)
        summary.add_column(style="bold cyan", width=16)
        summary.add_column()
        summary.add_row("Status", self._status)
        summary.add_row("Current", self._current_config)
        best = "–" if self._best_throughput is None else f"{self._best_throughput:.2f} req/s"
        summary.add_row("Best throughput", Text(best, style="bold green" if self._best_throughput is not None else "dim"))

        events = Group(*self._events) if self._events else Text("Waiting for activity…", style="dim")
        guidellm = (
            Group(*self._guidellm_output)
            if self._guidellm_output
            else Text("GuideLLM starts when the engine is ready.", style="dim")
        )
        return Group(
            Panel(summary, title="[bold]Auto-tune[/bold]", border_style="cyan"),
            self.progress,
            Panel(events, title="[bold]Events[/bold]", border_style="blue"),
            Panel(guidellm, title="[bold magenta]GuideLLM output[/bold magenta]", border_style="magenta"),
        )

    def _refresh(self) -> None:
        if self._started:
            self._live.update(self._render(), refresh=True)
