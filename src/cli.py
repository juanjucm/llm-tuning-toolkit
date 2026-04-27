import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from auto_tune.tuner import AutoTuner
from cli.auto_tune import tuner_app

HF_TOKEN = os.getenv("HF_TOKEN", "")

app = typer.Typer(
    name="llm-tuning-toolkit",
    help="A toolkit for automatic tuning and benchmarking of LLM serving configurations.",
    no_args_is_help=True,
)

app.command(name="tune", no_args_is_help=True)
def auto_tune(
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
    """Performs auto-tuning process for the provided configuration."""
    if not os.path.exists(config):
        print(f"Error: Configuration file not found: {config}")
        raise typer.Exit(code=1)

    tuner = AutoTuner(config, result_dir, dataset_id, cache_dir, hf_token)
    tuner.run_auto_tune()
