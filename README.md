# LLM Tunning Toolkit ⚙️

Toolkit for automatic tuning and benchmarking of LLM serving configurations.

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

This module provides a way to automatically detect the best LLM serving configuration that maximises throughput while being complient with a set of defined goodput criteria.

For running the script, make sure to provide a valid config yaml. Take a loot at `auto-tune-config.yaml` to check the format and expected parameters.

```concole
usage: uv run auto-tune [-h] [--config CONFIG] [--result-dir RESULT_DIR]

Auto-tune tool for finding optimal engine parameters.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to auto-tune configuration file
  --result-dir RESULT_DIR
                        Directory to save tuning results
```

## Multi Benchmarking Usage

**NOTE: still work in progress, you can expect failures.**

```console
usage: uv run multi-benchmarker [-h] [--config CONFIG] [--scenarios SCENARIOS] [--engines ENGINES] [--show-logs] [--save-dir SAVE_DIR]

Launch benchmarks based on a configuration file

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to benchmark configuration file
  --scenarios SCENARIOS
                        Specific scenarios to run, comma separated (i.e: "s1,s2,s3") (if not specified, runs all scenarios)
  --engines ENGINES     Specific engines to test, comma separated (i.e: "e1,e2,e3") (if not specified, tests all engines)
  --save-dir SAVE_DIR   Directory to save benchmark results
  --show-logs           Show engine container logs.
  ```

## Dashboard Usage

**NOTE: still work in progress, you can expect failures.**

```console
Usage: dashboard [OPTIONS]

Options:
  --from-results-dir TEXT  Load inference-benchmarker results from a directory
  --datasource TEXT        Load a Parquet file already generated
  --port INTEGER           Port to run the dashboard
  --help                   Show this message and exit.
```
