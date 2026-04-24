import typer

from cli.auto_tune import tuner_app

app = typer.Typer(
    name="llm-tuning-toolkit",
    help="A toolkit for automatic tuning and benchmarking of LLM serving configurations.",
    no_args_is_help=True,
)

app.add_typer(tuner_app)
