import argparse
import os
import sys
from pathlib import Path

from auto_tune.display import AutoTuneDisplay
from auto_tune.tuner import AutoTuner

HF_TOKEN = os.getenv("HF_TOKEN", "")

parser = argparse.ArgumentParser(description="Auto-tune tool for finding optimal engine parameters.")
parser.add_argument("--config", help="Path to auto-tune configuration file", required=True)
parser.add_argument("--result-dir", default="", help="Directory to save tuning results")
parser.add_argument("--dataset-id", help="Huggingface dataset where to dump resutls")
parser.add_argument(
    "--cache-dir",
    default=str(Path.home() / ".cache" / "huggingface" / "hub"),
    help="Cache directory for Huggingface models and datasets.",
)
parser.add_argument(
    "--hf-token", default=HF_TOKEN, help="Huggingface token to use for accesing models and dataset."
)
parser.add_argument(
    "--no-ui",
    action="store_true",
    help="Disable the Rich live display and use normal log output instead.",
)


def main() -> None:
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    display = None if args.no_ui else AutoTuneDisplay()
    tuner = AutoTuner(args.config, args.result_dir, args.dataset_id, args.cache_dir, args.hf_token, display)
    try:
        tuner.run_auto_tune()
    finally:
        if display:
            display.stop()
