# LLM Tuning Toolkit ⚙️

Toolkit for automatic tuning and benchmarking of LLM serving configurations.

> [!WARNING]
> Still work in progress, you can expect failures.
> Please, check TODO.md for WIP.

## Prerequisites

Auto-tuning uses [GuideLLM](https://github.com/vllm-project/guidellm) against
the OpenAI-compatible endpoint exposed by each engine container. It is installed
with the project dependencies.

GuideLLM keeps its normal terminal output during auto-tuning, including its live
benchmark progress. The tuner logs are shown before and after each GuideLLM run.

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

For a complete configuration, start with
[`examples/guidellm-auto-tune.yaml`](examples/guidellm-auto-tune.yaml). The
`scenario.data` list maps directly to GuideLLM `--data` descriptors, so it can
use synthetic text, local/Hugging Face datasets, trace replay, and multimodal
data. The tuner runs GuideLLM's throughput profile for each engine sweep, then
uses a constant-rate profile at progressively lower rates until every configured
SLO is met.

Set `scenario.load.kind` to `throughput` with `max_concurrency` for capacity
search, or to `concurrent` with `streams` for a fixed number of continuously
active users. Throughput mode uses the SLO-driven rate-reduction search;
concurrent mode evaluates each engine configuration at the declared stream count.

SLO names use `min_` or `max_` followed by a normalized metric, such as
`min_success_rate`, `max_ttft_p99_ms`, `max_e2e_p99_ms`, or
`min_output_tokens_per_second`. Results include raw GuideLLM JSON reports plus
`auto_tune_results.json` with the selected deployment configuration.

For tool calling, provide a GuideLLM dataset with tool-call messages and add the
`tool_calling_message_extractor` data preprocessor through
`scenario.guidellm_options.arguments`. Enable the matching vLLM tool parser in
the engine's `base_args`. For vision/audio/video, pass GuideLLM multimodal data
descriptors in the same `scenario.data` list.

```console
usage: uv run auto-tune [-h] [--config <config.yaml>] [--result-dir <result_dir>] [--dataset-id <dataset_id>] [--hf-token <hf_token>]

Auto-tune tool for finding optimal engine parameters.

options:
  -h, --help    show this help message and exit
  --config      Path to GuideLLM auto-tune configuration file
  --result-dir (optional) Directory to save tuning results
  --dataset-id (optional) Huggingface dataset where to dump results
  --hf-token   (optional) Huggingface token to use for accesing models and dataset.
```

## Multi Benchmarking Usage

This tool allows for easily define and launch benchmarking scenarios for a set of defined LLM runtimes with specified parameters.

For running the script, make sure to provide a valid config yaml. Take a loot at `bench_config.yaml` to check the format and expected parameters.

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

This tool launches a dashboard for visualizing benchmarking results.

```console
Usage: dashboard [OPTIONS]

Options:
  --from-results-dir TEXT  Load inference-benchmarker results from a directory
  --datasource TEXT        Load a Parquet file already generated
  --port INTEGER           Port to run the dashboard
  --help                   Show this message and exit.
```
