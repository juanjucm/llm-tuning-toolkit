"""Command-line interface for caller-managed GuideLLM benchmark suites."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .runner import BenchmarkSuite

app = typer.Typer(
    name="recipe-benchmark",
    add_completion=False,
    help="Benchmark an already deployed serving recipe with a reusable GuideLLM suite.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command()
def benchmark(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the benchmark-only YAML suite.",
        ),
    ],
    recipe: Annotated[
        str,
        typer.Option("--recipe", help="Caller-owned recipe identifier recorded with the results."),
    ],
    target: Annotated[
        str,
        typer.Option("--target", help="OpenAI-compatible endpoint for the already deployed recipe."),
    ],
    model: Annotated[
        str,
        typer.Option("--model", help="Model served by the target endpoint."),
    ],
    result_dir: Annotated[
        Path,
        typer.Option("--result-dir", file_okay=False, help="Directory in which to save benchmark results."),
    ] = Path("out"),
    context_window: Annotated[
        int | None,
        typer.Option(
            "--context-window",
            min=1,
            help="Model context window used by suite selection rules.",
        ),
    ] = None,
    capability: Annotated[
        list[str] | None,
        typer.Option(
            "--capability",
            help="Model capability used by suite selection rules; repeat for multiple values.",
        ),
    ] = None,
    selected_benchmark: Annotated[
        list[str] | None,
        typer.Option(
            "--benchmark",
            help="Run only this benchmark name; repeat for multiple values.",
        ),
    ] = None,
    include_tag: Annotated[
        list[str] | None,
        typer.Option("--include-tag", help="Run cases matching this tag; repeat for multiple values."),
    ] = None,
    exclude_tag: Annotated[
        list[str] | None,
        typer.Option("--exclude-tag", help="Skip cases matching this tag; repeat for multiple values."),
    ] = None,
    guidellm_command: Annotated[
        str,
        typer.Option("--guidellm-command", help="Path or name of the GuideLLM executable."),
    ] = "guidellm",
    no_console: Annotated[
        bool,
        typer.Option(
            "--no-console",
            help="Suppress GuideLLM's live console while retaining suite progress logs.",
        ),
    ] = False,
) -> None:
    """Run the selected suite entries against a caller-managed endpoint."""
    summary = BenchmarkSuite(
        config_path=str(config),
        recipe=recipe,
        target=target,
        model=model,
        result_dir=str(result_dir),
        context_window=context_window,
        capabilities=capability or (),
        benchmark_names=selected_benchmark or (),
        include_tags=include_tag or (),
        exclude_tags=exclude_tag or (),
        guidellm_command=guidellm_command,
        show_console=not no_console,
    ).run()
    if any(item["status"] == "failed" for item in summary["benchmarks"]):
        raise typer.Exit(1)


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
