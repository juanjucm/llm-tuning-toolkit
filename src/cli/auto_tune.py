import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer

from auto_tune.tuner import AutoTuner

HF_TOKEN = os.getenv("HF_TOKEN", "")

tuner_app = typer.Typer(
    name="auto-tune",
    help="Auto-tune tool for finding serving parameters that maximise defined SLOs.",
    no_args_is_help=True,
)


@tuner_app.command()
def run(
    config: Annotated[str, typer.Option(help="Path to auto-tune configuration file")],
    result_dir: Annotated[str, typer.Option(help="Directory to save tuning results")] = "",
    dataset_id: Annotated[Optional[str], typer.Option(help="Huggingface dataset where to dump results")] = None,
    cache_dir: Annotated[
        str,
        typer.Option(help="Cache directory for Huggingface models and datasets."),
    ] = str(Path.home() / ".cache" / "huggingface" / "hub"),
    hf_token: Annotated[
        str,
        typer.Option(help="Huggingface token for accessing models and datasets."),
    ] = HF_TOKEN,
) -> None:
    """Run the auto-tuning process based on the provided configuration."""
    if not os.path.exists(config):
        print(f"Error: Configuration file not found: {config}")
        raise typer.Exit(code=1)

    tuner = AutoTuner(config, result_dir, dataset_id, cache_dir, hf_token)
    tuner.run_auto_tune()
