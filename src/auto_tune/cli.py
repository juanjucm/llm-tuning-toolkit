import argparse
import os
from pathlib import Path
import sys

from auto_tune.display import AutoTuneDisplay
from auto_tune.tuner import AutoTuner

HF_TOKEN = os.getenv("HF_TOKEN", "")

parser = argparse.ArgumentParser(description="Auto-tune tool for finding optimal engine parameters.")
parser.add_argument("--config", help="Path to auto-tune configuration file", required=True)
parser.add_argument("--result-dir", default="", help="Directory to save tuning results")
parser.add_argument("--dataset-id", help="Hugging Face dataset where results are uploaded")
parser.add_argument(
    "--cache-dir",
    default=str(Path.home() / ".cache" / "huggingface" / "hub"),
    help="Cache directory for Hugging Face models and datasets.",
)
parser.add_argument(
    "--hf-token", default=HF_TOKEN, help="Hugging Face token used for models, datasets, and Trackio Spaces."
)
parser.add_argument(
    "--no-ui",
    action="store_true",
    help="Disable the Rich live display and use normal log output instead.",
)
parser.add_argument("--trackio-project", help="Trackio project name. Providing it enables metric tracking.")
trackio_destination = parser.add_mutually_exclusive_group()
trackio_destination.add_argument("--trackio-space-id", help="Hugging Face Space used for Trackio metrics.")
trackio_destination.add_argument("--trackio-server-url", help="Self-hosted Trackio server write-access URL.")
parser.add_argument("--trackio-group", help="Optional Trackio run group. Defaults to the scenario name.")


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
    if not args.trackio_project and any((args.trackio_space_id, args.trackio_server_url, args.trackio_group)):
        parser.error("--trackio-project is required when other Trackio options are provided")

    tuner = AutoTuner(
        config_path=args.config,
        result_dir=args.result_dir,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        display=display,
        trackio_project=args.trackio_project,
        trackio_space_id=args.trackio_space_id,
        trackio_server_url=args.trackio_server_url,
        trackio_group=args.trackio_group,
    )
    tuner.run_auto_tune()
