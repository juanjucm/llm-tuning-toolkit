"""Optional Trackio integration for benchmark metrics."""

from __future__ import annotations

import importlib
import logging
from typing import Any


class TrackioTracker:
    """Manage one optional Trackio run at a time without affecting benchmarks."""

    def __init__(self, config: dict[str, Any] | None, logger: logging.Logger) -> None:
        self.config = config or {}
        self.logger = logger
        self.enabled = config is not None and self.config.get("enabled", True)
        self._trackio: Any = None
        self._active = False

    def start_run(self, *, name: str, group: str, config: dict[str, Any]) -> None:
        if not self.enabled:
            return

        try:
            if self._trackio is None:
                self._trackio = importlib.import_module("trackio")

            init_args = {
                "project": self.config.get("project", "llm-tuning-toolkit"),
                "name": name,
                "group": self.config.get("group", group),
                "config": config,
                "auto_log_cpu": False,
                "auto_log_gpu": False,
            }
            for key in ("space_id", "server_url"):
                if self.config.get(key):
                    init_args[key] = self.config[key]

            self._trackio.init(**init_args)
            self._active = True
        except Exception as error:
            self._active = False
            self.logger.warning("Could not start Trackio run: %s", error)

    def log_metrics(
        self,
        metrics: dict[str, Any],
        *,
        slo_met: bool,
        target_rate: float | None = None,
    ) -> None:
        if not self._active:
            return

        values = dict(metrics)
        values["slo_met"] = int(slo_met)
        values["rate_benchmark"] = int(target_rate is not None)
        if target_rate is not None:
            values["target_rate"] = target_rate

        try:
            self._trackio.log(values)
        except Exception as error:
            self.logger.warning("Could not log metrics to Trackio: %s", error)

    def finish_run(self) -> None:
        if not self._active:
            return

        try:
            self._trackio.finish()
        except Exception as error:
            self.logger.warning("Could not finish Trackio run: %s", error)
        finally:
            self._active = False
