import argparse
import os
import sys

from auto_tune import AutoTuner

parser = argparse.ArgumentParser(description="Auto-tune tool for finding optimal engine parameters.")
parser.add_argument("--config", default="./auto-tune-config.yaml", help="Path to auto-tune configuration file")
parser.add_argument("--result-dir", default="./auto_tune_results", help="Directory to save tuning results")


def main() -> None:
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    tuner = AutoTuner(args.config, args.result_dir)
    tuner.run_auto_tune()
