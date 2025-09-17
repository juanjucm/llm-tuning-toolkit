# Benchmarker ⚙️

Evaluate LLM serving engines for a certain LLM based on different benchmarking scenarios.

## Prerequisites

This project requires `inference-benchmarker`. Install it using:

```bash
cargo install --git https://github.com/juanjucm/inference-benchmarker/
```

## Installation

First you need to setup your environment with [`uv`](https://github.com/astral-sh/uv),
or with your preferred Python environment manager.

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install .
```

## Usage

```console
usage: benchmarker [-h] [--config CONFIG] [--scenarios SCENARIOS] [--engines ENGINES] [--show-logs] [--save-dir SAVE_DIR]

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