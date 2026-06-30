# LLM Tuning Toolkit ⚙️

Toolkit for automatic tuning and benchmarking of LLM serving configurations.

> [!WARNING]
> Still work in progress, you can expect failures.
> Please, check TODO.md for WIP.

## Prerequisites

The default benchmark backend is `inference-benchmarker`. Install it with:

```bash
cargo install --git https://github.com/juanjucm/inference-benchmarker/
```

The benchmark executor is pluggable. `vllm-bench` can be selected with
`--benchmark-backend vllm-bench`, or a custom command can be passed with
`--benchmark-command`. Auto-tune metric extraction currently supports
`inference-benchmarker` result JSON.

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

## Configs

Reusable configs live under `configs/`.

```text
configs/
  benchmarks/<model>/<instance>/bench_config.yaml
  auto-tune/<model>/<instance>/<scenario>.yaml
```

Generated result folders and Parquet files are local outputs and ignored by git. Auto-tune still copies the config used for a run into its result folder for reproducibility.

## Auto Tuning Usage

This module automatically detects the best LLM serving configuration that maximizes throughput while staying compliant with a set of defined goodput criteria.

For running the script, make sure to provide a valid config yaml. Take a look at `configs/auto-tune/.../<scenario>.yaml` to check the format and expected parameters.

```console
usage: uv run auto-tune --config <config.yaml> [--result-dir <result_dir>] [--dataset-id <dataset_id>] [--cache-dir <cache_dir>] [--hf-token <hf_token>] [--benchmark-backend <backend>] [--benchmark-command <cmd>] [--verbose]

Tune serving-engine parameters by running benchmark backends.

options:
  -h, --help              show this help message and exit
  --config                Path to auto-tune YAML configuration
  --result-dir            Directory to save tuning results
  --dataset-id            Hugging Face dataset where results should be uploaded
  --cache-dir             Cache directory for Hugging Face models and datasets
  --hf-token              Hugging Face token for model and dataset access
  --benchmark-backend     Benchmark executor backend: inference-benchmarker or vllm-bench
  --benchmark-command     Override benchmark command, e.g. "python -m my_bench"
  --verbose               Enable DEBUG logging
```

## Multi Benchmarking Usage

This tool defines and launches benchmark scenarios for a set of LLM runtimes with specified parameters.

For running the script, make sure to provide a valid config yaml. Take a look at `configs/benchmarks/.../bench_config.yaml` to check the format and expected parameters.

```console
usage: uv run multi-benchmarker [-h] --config CONFIG [--scenarios SCENARIOS] [--engines ENGINES] [--output-path OUTPUT_PATH] [--benchmark-backend BACKEND] [--benchmark-command CMD] [--cache-dir CACHE_DIR] [--hf-token HF_TOKEN] [--show-logs] [--verbose]

Run configured serving-engine benchmarks with a selectable benchmark backend.

options:
  -h, --help                 show this help message and exit
  --config CONFIG            Path to benchmark YAML config
  --scenarios SCENARIOS      Specific scenarios to run, comma separated
  --engines ENGINES          Specific engines to test, comma separated
  --output-path OUTPUT_PATH  Directory to save benchmark results
  --benchmark-backend        Benchmark executor backend: inference-benchmarker or vllm-bench
  --benchmark-command        Override benchmark command
  --cache-dir CACHE_DIR      Cache directory for Hugging Face models and datasets
  --hf-token HF_TOKEN        Hugging Face token for model access
  --show-logs                Stream engine container logs while running
  --verbose                  Enable DEBUG logging
```

## Dashboard Usage

This tool launches a dashboard for visualizing benchmarking results.
Provide either a results directory or a Parquet datasource.

```console
Usage: dashboard [OPTIONS]

Options:
  --from-results-dir TEXT  Load inference-benchmarker results from a directory
  --datasource TEXT        Load a Parquet file already generated
  --port INTEGER           Port to run the dashboard
  --help                   Show this message and exit.
```
