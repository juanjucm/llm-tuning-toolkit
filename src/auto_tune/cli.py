"""Command-line interface for the auto-tuning workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from auto_tune.display import AutoTuneDisplay
from auto_tune.tuner import AutoTuner

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"

app = typer.Typer(
    name="auto-tune",
    add_completion=False,
    help="Find the highest-throughput engine configuration that satisfies a scenario's SLOs.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command()
def tune(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            help="Path to the auto-tune YAML configuration.",
        ),
    ],
    result_dir: Annotated[
        Path,
        typer.Option("--result-dir", file_okay=False, help="Directory in which to save tuning results."),
    ] = Path("out"),
    dataset_id: Annotated[
        str | None,
        typer.Option("--dataset-id", help="Hugging Face dataset to which results are uploaded."),
    ] = None,
    cache_dir: Annotated[
        Path,
        typer.Option("--cache-dir", file_okay=False, help="Hugging Face model and dataset cache."),
    ] = DEFAULT_CACHE_DIR,
    hf_token: Annotated[
        str,
        typer.Option("--hf-token", envvar="HF_TOKEN", help="Hugging Face token for Hub and Trackio access."),
    ] = "",
    no_ui: Annotated[
        bool,
        typer.Option("--no-ui", help="Run without terminal UI or benchmark log output."),
    ] = False,
    trackio_project: Annotated[
        str | None,
        typer.Option("--trackio-project", help="Trackio project name; providing it enables tracking."),
    ] = None,
    trackio_space_id: Annotated[
        str | None,
        typer.Option("--trackio-space-id", help="Hugging Face Space used for Trackio metrics."),
    ] = None,
    trackio_server_url: Annotated[
        str | None,
        typer.Option("--trackio-server-url", help="Self-hosted Trackio server write-access URL."),
    ] = None,
    trackio_group: Annotated[
        str | None,
        typer.Option("--trackio-group", help="Trackio run group; defaults to the scenario name."),
    ] = None,
) -> None:
    """Run all engine configurations defined in CONFIG."""
    if trackio_space_id and trackio_server_url:
        raise typer.BadParameter(
            "use either --trackio-space-id or --trackio-server-url, not both",
            param_hint="Trackio destination",
        )
    if not trackio_project and any((trackio_space_id, trackio_server_url, trackio_group)):
        raise typer.BadParameter(
            "--trackio-project is required when other Trackio options are provided",
            param_hint="--trackio-project",
        )

    display = None if no_ui else AutoTuneDisplay()
    tuner_options = {
        "config_path": str(config),
        "result_dir": str(result_dir),
        "dataset_id": dataset_id,
        "cache_dir": str(cache_dir),
        "hf_token": hf_token or os.getenv("HF_TOKEN", ""),
        "trackio_project": trackio_project,
        "trackio_space_id": trackio_space_id,
        "trackio_server_url": trackio_server_url,
        "trackio_group": trackio_group,
        "quiet": no_ui,
    }
    if display is not None:
        tuner_options["display"] = display

    AutoTuner(**tuner_options).run_auto_tune()


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
