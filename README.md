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

For data descriptor fields, preprocessors, and dataset formats, refer to the
[GuideLLM datasets guide](https://github.com/vllm-project/guidellm/blob/main/docs/guides/datasets.md).
Auto-tune owns the load profile, constraints, output location, and SLO retry policy.

The supported `scenario.load.kind` values are `throughput`, `concurrent`,
`constant`, `poisson`, and `replay`. Throughput mode uses the SLO-driven
constant-rate fallback; all other modes evaluate the declared workload directly.

### How to simulate your scenario

Choose the traffic model separately from the condition that ends a benchmark.
For example, a fixed request count is an end condition; it can be used with a
concurrent-user, fixed-rate, or trace-replay workload.

#### Capacity discovery: maximum demand up to a concurrency limit

Use this to find the deployment configuration with the greatest completed
throughput, while limiting the number of in-flight requests. It is a stress / capacity
test, not a model of a fixed user population: GuideLLM continually issues work as fast
as possible until it reaches `max_concurrency`.

```yaml
scenario:
  load:
    kind: throughput
    max_concurrency: 512
```

This is the default auto-tuning strategy and enables the SLO-driven rate fallback.

#### Fixed active users: closed-loop traffic

Use this when the scenario defines a known number of active clients. Each stream sends
its next request after the prior one completes, so `streams` represents continuously
active clients rather than requests per second.

```yaml
scenario:
  load:
    kind: concurrent
    streams: 512
```

Without a delay this models automated clients that immediately send another request.
For interactive chat or agent sessions, use multi-turn data and synthetic `delay`
(think time) between turns:

```yaml
scenario:
  data:
    - kind: synthetic_text
      prompt_tokens: 1024
      output_tokens: 256
      turns: 3
      delay: 5
      delay_min: 2
      delay_max: 12
  load:
    kind: concurrent
    streams: 512
```

#### Open-arrival traffic: independent requests

For traffic where clients arrive independently, use `constant` or `poisson`.
`constant` sends an even request rate; `poisson` introduces natural variation around
an average rate and is usually the closer production model. A useful first estimate is
`arrival_rate ≈ active_users / (mean_response_time + mean_think_time)`.

```yaml
scenario:
  load:
    kind: poisson             # or constant
    rate: 64                  # requests per second
    max_concurrency: 512      # optional safety cap
```

#### Replay real traffic

When production traces are available, GuideLLM's `replay` profile can reproduce their
timestamped arrivals. This is the most faithful workload model.

```yaml
scenario:
  data:
    - kind: trace_synthetic
      path: /absolute/path/traffic.jsonl
  load:
    kind: replay
    time_scale: 1.0
```

#### Ending a benchmark cleanly

`scenario.constraints` maps directly to repeated GuideLLM `--constraint` descriptors.
Use one primary completion boundary. `max_duration` gives a fixed wall-clock test but
cancels in-flight work at the deadline; `max_requests` gives a fixed request-count test
and lets created requests reach a terminal state.

```yaml
scenario:
  # Capacity discovery for 60 seconds.
  constraints:
    - kind: max_duration
      seconds: 60
```

```yaml
scenario:
  # Fixed-user or fixed-rate test: 10,000 created requests reach a terminal outcome.
  constraints:
    - kind: max_requests
      count: 10000
```

When throughput mode needs its SLO rate fallback, `rate_constraints` controls the
fallback attempts. If `constraints` is explicitly set and `rate_constraints` is absent,
the same constraints are used for both. If neither constraint list nor legacy duration
setting is supplied, auto-tune defaults to `max_requests: 1000` for both runs. Set
`max_requests` and `rate_max_requests` to change those defaults. Existing
`throughput_duration_seconds` and `rate_duration_seconds` configurations remain
supported as explicit legacy duration constraints.

Do not pass `constraint` through `scenario.guidellm_options.arguments`: auto-tune rejects
it to prevent conflicting stop conditions.

GuideLLM also provides `max_errors`, `max_error_rate`, `max_global_error_rate`, and
`over_saturation` constraints. They are useful safety / early-exit conditions alongside a
primary duration or request-count boundary. For example:

```yaml
scenario:
  constraints:
    - kind: max_requests
      count: 10000
    - kind: max_error_rate
      rate: 0.02
    - kind: over_saturation
      mode: enforce
      min_seconds: 30
```

When using duration-based tests, keep `errored` and `incomplete` request rates separate:
an error is a terminal backend/request failure, while an incomplete request can be a
benchmark cut-off cancellation.

SLO names use `min_` or `max_` followed by a normalized metric, such as
`min_success_rate`, `max_ttft_p99_ms`, `max_e2e_p99_ms`, or
`min_output_tokens_per_second`. Results include raw GuideLLM JSON reports plus
`auto_tune_results.json` with the selected deployment configuration.

### Track benchmark metrics with Trackio

Auto-tuning can optionally send its normalized benchmark metrics to
[Trackio](https://github.com/gradio-app/trackio). Each engine-parameter
combination is recorded as a separate Trackio run. The primary benchmark and
any fixed-rate SLO fallback attempts are logged as successive steps in that run.

For local storage, configure only a project name:

```yaml
trackio:
  project: llm-tuning-toolkit
```

After the auto-tune run, open the local dashboard with:

```bash
trackio show --project llm-tuning-toolkit
```

To log to a Hugging Face Space, add `space_id`:

```yaml
trackio:
  project: llm-tuning-toolkit
  space_id: username/trackio
```

To log to a self-hosted Trackio server, use its write-access URL. The write
token can be included in that URL or, preferably, supplied through the
`TRACKIO_WRITE_TOKEN` environment variable:

```yaml
trackio:
  project: llm-tuning-toolkit
  server_url: http://trackio.example:7860
```

Start a local network-accessible server with `trackio show --host 0.0.0.0`.
`space_id` and `server_url` are mutually exclusive. Set `enabled: false` to
temporarily disable a configured integration.

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
