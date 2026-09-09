# LLM Tuning Toolkit ⚙️

Toolkit for automatic tuning and benchmarking of LLM serving configurations.

> [!WARNING]
> Still work in progress, you can expect failures.
> Please, check TODO.md for WIP.

## Prerequisites

Auto-tuning uses [GuideLLM](https://github.com/vllm-project/guidellm) against
the OpenAI-compatible endpoint exposed by each engine container. It is installed
with the project dependencies.

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
## Benchmark an already deployed recipe

Use `recipe-benchmark` when a workflow has already selected and deployed a
serving recipe. This path does not read engine arguments, start containers, or
search for a better recipe. The caller supplies:

- a recipe identifier for result attribution;
- the OpenAI-compatible target URL;
- the served model;
- optional model traits used to select applicable workloads.

The YAML file contains only reusable GuideLLM benchmark policy. Target and model
fields are rejected in the suite so a checked-in suite cannot silently redirect
a workflow.

For a private Hugging Face endpoint, expose `HF_TOKEN` to the job:

```bash
export HF_TOKEN=hf_...
uv run recipe-benchmark ...
```

The benchmark runner reads `HF_TOKEN` and sets GuideLLM's `openai_http`
`api_key`; GuideLLM therefore sends `Authorization: Bearer <HF_TOKEN>` for both
backend validation and benchmark requests. The token is never written to
`benchmark_results.json`. `target`, `model`, and `api_key` are runtime-owned and
are rejected if they appear in the suite YAML.

[`examples/guidellm-recipe-benchmark.yaml`](examples/guidellm-recipe-benchmark.yaml)
is the production suite. It covers:

- single-stream latency baseline;
- closed-loop concurrency from 1 to 128 active streams;
- an open-loop Poisson capacity sweep;
- long-context prefill;
- decode-heavy generation;
- extended reasoning generation when the caller declares `reasoning`; and
- image-plus-text concurrency when the caller declares `vision`.

Every load profile has deterministic randomization, warmup/cooldown exclusion,
bounded request counts, early error-rate termination, and success-rate
evaluation. Raw GuideLLM reports remain the metric source of truth.

Run the core text suite against an existing deployment:

```bash
uv run recipe-benchmark \
  --config examples/guidellm-recipe-benchmark.yaml \
  --recipe vllm-l40s-fp8 \
  --target http://serving.internal:8000 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --context-window 131072 \
  --result-dir out
```

Declare model capabilities to enable applicable optional cases:

```bash
uv run recipe-benchmark \
  --config examples/guidellm-recipe-benchmark.yaml \
  --recipe vllm-qwen-vl \
  --target http://serving.internal:8000 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --context-window 32768 \
  --capability vision
```

`--benchmark` selects exact case names. `--include-tag` and `--exclude-tag`
filter groups such as `core`, `long-context`, or `optional`; each option is
repeatable. Unknown benchmark names and an entirely empty selection fail before
GuideLLM starts.

Model-aware selection is declared per benchmark:

```yaml
- name: long-context-prefill
  description: Prefill latency with 16K-28K token prompts.
  tags: [core, long-context, prefill]
  selection:
    min_context_window: 32768
    capabilities: [text]       # optional
    model_patterns: ["org/*"]  # optional glob
    excluded_model_patterns: ["*/legacy-*"]
  data: [...]
  profile: {kind: concurrent, streams: [1, 2, 4, 8]}
  constraints: [{kind: max_requests, count: 100}]
```

If a case requires a context window or capability that the caller did not
provide, the case is recorded as `skipped` with the exact reason. `selection.enabled:
false` statically disables a case.

The same contract is available as a Python API:

```python
from auto_tune.benchmarking import BenchmarkSuite

summary = BenchmarkSuite(
    "examples/guidellm-recipe-benchmark.yaml",
    recipe="vllm-l40s-fp8",
    target="http://serving.internal:8000",
    model="meta-llama/Llama-3.1-8B-Instruct",
    context_window=131072,
    capabilities={"reasoning"},
    exclude_tags={"vision"},
).run()
```

Each run stores the copied suite, one raw GuideLLM JSON report per executed
case, and `benchmark_results.json` under:

```text
out/<suite>/<recipe>/run_<timestamp>_<id>/
```

The consolidated report records caller inputs, executed and skipped cases,
concrete load points, normalized metrics, and optional SLO verdicts. One case
failing does not suppress later cases; the CLI exits non-zero after recording
the complete result. SLO misses remain benchmark data and do not change the exit
code.

Benchmark code lives under `auto_tune.benchmarking`: typed suite configuration
and selection in `config.py`, execution/reporting in `runner.py`, and workflow
arguments in `cli.py`. It shares GuideLLM normalization and SLO evaluation with
auto-tune, but no deployment lifecycle or tuning policy.

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

### Engine backends

Auto-tune can launch vLLM or SGLang containers. Engine `base_args`,
`value_args_pool`, and `action_args_pool` are passed through, so use the native
server arguments for the selected engine. Set `engine.kind: sglang` to use
SGLang's `python3 -m sglang.launch_server` entrypoint and `/health_generate`
readiness check; both can be overridden with `engine.entrypoint` and
`engine.health_path`.

The special `tp-dp-combinations` sweep maps to `--tensor-parallel-size` and
`--data-parallel-size` for vLLM, and to `--tp-size` and `--dp-size` for SGLang.
See [`examples/guidellm-sglang-auto-tune.yaml`](examples/guidellm-sglang-auto-tune.yaml)
for a runnable single-GPU configuration. SGLang's official
[Docker installation guide](https://docs.sglang.io/docs/get-started/install) and
[server arguments](https://docs.sglang.io/docs/advanced_features/server_arguments)
describe the engine-specific options available for additional sweeps.

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
Trackio options are supplied through the CLI, not the benchmark YAML.

For local storage, provide only a project name:

```bash
uv run auto-tune --config examples/guidellm-auto-tune.yaml \
  --trackio-project llm-tuning-toolkit
```

After the auto-tune run, open the local dashboard with:

```bash
trackio show --project llm-tuning-toolkit
```

To log to a Hugging Face Space, add its ID. `--hf-token` is forwarded to
Trackio for Space authentication; it defaults to the `HF_TOKEN` environment
variable:

```bash
uv run auto-tune --config examples/guidellm-auto-tune.yaml \
  --trackio-project llm-tuning-toolkit \
  --trackio-space-id username/trackio \
  --hf-token "$HF_TOKEN"
```

To log to a self-hosted Trackio server, use its write-access URL. The write
token can be included in that URL or, preferably, supplied through the
`TRACKIO_WRITE_TOKEN` environment variable:

```bash
uv run auto-tune --config examples/guidellm-auto-tune.yaml \
  --trackio-project llm-tuning-toolkit \
  --trackio-server-url http://trackio.example:7860
```

Start a local network-accessible server with `trackio show --host 0.0.0.0`.
`--trackio-space-id` and `--trackio-server-url` are mutually exclusive. Omit
`--trackio-project` to disable metric tracking. Use `--trackio-group` to
override the scenario name used to group runs.

For tool calling, provide a GuideLLM dataset with tool-call messages and add the
`tool_calling_message_extractor` data preprocessor through
`scenario.guidellm_options.arguments`. Two further settings are mandatory on
GuideLLM 0.7.3, and omitting either one silently benchmarks plain chat at 100%
"success" instead of tool calls:

- `load_kwargs: {split: train}` on the data descriptor — required on guidellm
  0.7.3: file loaders return a `DatasetDict` and the column mapper crashes
  without an explicit split.
- An explicit `data-column-mapper` with
  `column_mappings: {text_column: messages, tools_column: tools}` — `messages` is
  not a default prompt column name, and `column_mappings` replaces the defaults
  wholesale, so `tools` must be listed too. Without `tools_column` no tool
  definitions reach the server.

Requests must also use `backend.request_format: /v1/chat/completions`, because
OpenAI tool definitions are carried in chat-completion requests. Enable the
matching vLLM tool parser in the engine's `base_args`.
[`examples/guidellm-tool-calls.yaml`](examples/guidellm-tool-calls.yaml) is a
complete, runnable configuration. For vision/audio/video, pass GuideLLM
multimodal data descriptors in the same `scenario.data` list.

The Typer-based CLI validates configuration paths and exposes its full option
reference through built-in help:

```bash
uv run auto-tune --help
uv run auto-tune --config examples/guidellm-auto-tune.yaml
```

The main options are `--result-dir`, `--dataset-id`, `--cache-dir`,
`--hf-token`, `--no-ui`, and the `--trackio-*` options described above.
`--hf-token` also reads `HF_TOKEN` when it is not supplied explicitly.
Results are stored under `out/` unless `--result-dir` specifies another path.

By default, auto-tune uses a colored static terminal view showing sweep progress,
the active parameter combination, the best valid throughput, and GuideLLM's live
benchmark progress. GuideLLM runs in a pseudo-terminal so its interactive progress can
be embedded without including setup messages or final report tables. Use `--no-ui`
to run without the terminal UI: no Rich live display is installed, GuideLLM's own
console output is suppressed, and the tuner's log records are written to stdout as
plain text.
