"""Pure formatting helpers for the Rich auto-tune display."""

from __future__ import annotations

from typing import Any

from rich.text import Text


def format_config(parameters: dict[str, Any]) -> str:
    value_args = parameters.get("value_args", {})
    action_args = [name for name, enabled in parameters.get("action_args", {}).items() if enabled]
    values = [f"{name}={value}" for name, value in value_args.items()]
    return ", ".join([*values, *action_args]) or "base arguments"


def format_workload(load: dict[str, Any]) -> str:
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


def format_constraints(constraints: object) -> str:
    if isinstance(constraints, (dict, str)):
        constraints = [constraints]
    if not isinstance(constraints, list) or not constraints:
        return "–"
    return " · ".join(format_constraint(item) for item in constraints)


def format_constraint(constraint: object) -> str:
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
    labels = {"over_saturation": "Over-saturation guard"}
    values = [str(value) for name, value in constraint.items() if name != "kind"]
    return ": ".join(filter(None, [labels.get(kind, kind.replace("_", " ").title()), ", ".join(values)]))


def format_data(data: object) -> str:
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


def format_slos(slos: object) -> list[Text]:
    if not isinstance(slos, dict):
        return []
    return [format_slo(name, value) for name, value in slos.items()]


def format_slo(name: str, value: object) -> Text:
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
        elif metric == "throughput":
            value_text = f"{value:g} req/s"
        elif metric in {"output_tokens_per_second", "total_tokens_per_second"}:
            value_text = f"{value:g} tok/s"
        else:
            value_text = f"{value:g}"
    else:
        value_text = str(value)
    return Text.assemble((f"• {label}", "cyan"), f"  {operator} {value_text}")


def format_metric(metrics: dict[str, float], name: str, unit: str) -> str:
    value = metrics.get(name)
    return "–" if value is None else f"{value:.2f} {unit}"


def format_percentage(value: float | None) -> str:
    return "–" if value is None else f"{value * 100:.1f}%"


def format_latency_pair(metrics: dict[str, float], prefix: str) -> str:
    average = metrics.get(f"{prefix}_avg_ms")
    p99 = metrics.get(f"{prefix}_p99_ms")
    if average is None and p99 is None:
        return "–"
    average_text = "–" if average is None else f"{average:.1f}"
    p99_text = "–" if p99 is None else f"{p99:.1f}"
    return f"{average_text} / {p99_text} ms"
