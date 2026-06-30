# LLM Tuning Toolkit ⚙️

Toolkit for automatic tuning and benchmarking of LLM serving configurations.

> [!WARNING]
> Still work in progress, you can expect failures.
> Please, check TODO.md for WIP.

## Prerequisites

This project requires `inference-benchmarker`. Install it using:

```bash
cargo install --git https://github.com/juanjucm/inference-benchmarker/
```

## Installation

First you need to setup your environment with [`uv`](https://github.com/astral-sh/uv).
```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install dependencies with -e for dev mode:

```bash
uv pip install -e .
```
## Auto Tuning Usage

This module provides a way to automatically detect the best LLM serving configuration that maximizes throughput while being compliant with a set of defined goodput criteria.

For running the script, make sure to provide a valid config yaml. Take a look at `auto-tune-config.yaml` to check the format and expected parameters.

```console
usage: uv run auto-tune [-h] [--config <config.yaml>] [--result-dir <result_dir>] [--dataset-id <dataset_id>] [--cache-dir <cache_dir>] [--hf-token <hf_token>]

Auto-tune tool for finding optimal engine parameters.

options:
  -h, --help    show this help message and exit
  --config      Path to auto-tune configuration file
  --result-dir (optional) Directory to save tuning results
  --dataset-id (optional) Huggingface dataset where to dump results
  --cache-dir  (optional) Cache directory for Hugging Face models and datasets
  --hf-token   (optional) Hugging Face token to use for accessing models and dataset.
```

## Multi Benchmarking Usage

This tool lets you define and launch benchmarking scenarios for a set of LLM runtimes with specified parameters.

For running the script, make sure to provide a valid config yaml. Take a look at `bench_config.yaml` to check the format and expected parameters.

```console
usage: uv run multi-benchmarker [-h] [--config CONFIG] [--scenarios SCENARIOS] [--engines ENGINES] [--output-path OUTPUT_PATH] [--show-logs]

Launch benchmarks based on a configuration file

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to benchmark configuration file
  --scenarios SCENARIOS
                        Specific scenarios to run, comma separated (i.e: "s1,s2,s3") (if not specified, runs all scenarios)
  --engines ENGINES     Specific engines to test, comma separated (i.e: "e1,e2,e3") (if not specified, tests all engines)
  --output-path OUTPUT_PATH
                        Directory to save benchmark results
  --show-logs           Show engine container logs.
```

## Dashboard Usage

This tool launches a dashboard for visualizing benchmarking results.

```console
Usage: dashboard [OPTIONS]

Options:
  --from-results-dir TEXT  Load inference-benchmarker results from a directory
  --datasource TEXT        Load a Parquet file already generated
  --port INTEGER           Port to run the dashboard
  --help                   Show this message and exit.
```
