"""SLO evaluation shared by tuning and fixed-recipe benchmarks."""

from __future__ import annotations

from typing import Any


def evaluate_slos(slos: dict[str, float], metrics: dict[str, float]) -> tuple[list[dict[str, Any]], bool]:
    """Compare normalized metrics against min_/max_ SLO thresholds."""
    results: list[dict[str, Any]] = []
    for threshold_name, threshold_value in slos.items():
        if not threshold_name.startswith(("min_", "max_")):
            raise ValueError(f"SLO '{threshold_name}' must start with min_ or max_")
        metric_name = threshold_name[4:]
        if metric_name not in metrics:
            raise ValueError(f"SLO '{threshold_name}' refers to unsupported metric '{metric_name}'")
        metric_value = metrics[metric_name]
        meets = (
            metric_value <= threshold_value
            if threshold_name.startswith("max_")
            else metric_value >= threshold_value
        )
        results.append({threshold_name: threshold_value, metric_name: metric_value, "meets": meets})
    return results, all(result["meets"] for result in results)
