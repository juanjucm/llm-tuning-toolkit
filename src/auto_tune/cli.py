import argparse
import logging
import os
import sys
from pathlib import Path

from auto_tune.tuner import AutoTuner

HF_TOKEN = os.getenv("HF_TOKEN", "")

parser = argparse.ArgumentParser(description="Tune serving-engine parameters by running benchmark backends.")
parser.add_argument("--config", help="Path to auto-tune YAML configuration", required=True)
parser.add_argument("--result-dir", default="", help="Directory to save tuning results")
parser.add_argument("--dataset-id", help="Hugging Face dataset where results should be uploaded")
parser.add_argument(
    "--cache-dir",
    default=str(Path.home() / ".cache" / "huggingface" / "hub"),
    help="Cache directory for Hugging Face models and datasets.",
)
parser.add_argument("--hf-token", default=HF_TOKEN, help="Hugging Face token for model and dataset access")
parser.add_argument(
    "--benchmark-backend",
    default="inference-benchmarker",
    choices=["inference-benchmarker", "vllm-bench"],
    help="Benchmark executor backend. Existing auto-tune metrics support inference-benchmarker output.",
)
parser.add_argument(
    "--benchmark-command",
    help="Override benchmark command, e.g. 'python -m my_bench'. Defaults depend on backend.",
)
parser.add_argument(
    "--verbose",
    action="store_true",
    help="Enable DEBUG logging, including benchmark subprocess output.",
)


def main() -> None:
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    tuner = AutoTuner(
        args.config,
        result_dir=args.result_dir,
        dataset_id=args.dataset_id,
        cache_dir=args.cache_dir,
        hf_token=args.hf_token,
        benchmark_backend=args.benchmark_backend,
        benchmark_command=args.benchmark_command,
        verbose=args.verbose,
    )
    tuner.run_auto_tune()
