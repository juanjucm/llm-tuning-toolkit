# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "vllm==0.11.0",
#   # Installing the project is what makes this job a real guard: the benchmark
#   # runs through auto_tune's own driver instead of a copy of it. This also
#   # pulls guidellm[recommended,vision] and huggingface_hub transitively.
#   # Point the ref at main once this PR merges.
#   "llm-tuning-toolkit @ git+https://github.com/juanjucm/llm-tuning-toolkit@migrate-to-guidellm",
# ]
# ///
"""
Serves Qwen/Qwen3-0.6B with vLLM, then runs:
  A. sharegpt   - ShareGPT json_file + tool_calling_message_extractor
  B. hf_dataset - garage-bAInd/Open-Platypus + column mapper
  C. trace      - trace_synthetic + replay profile
  D. rate       - fixed-rate rerun of A at 50% of measured throughput,
                  mirroring the tuner's SLO rate fallback

Prints real throughput / TTFT / ITL / E2E per pathway and evaluates the chat
SLOs from the GuideLLM SLO guide. Exit code reflects benchmark failures, not
SLO misses (an SLO miss is a result, not an error).

Dispatched by .github/workflows/real-engine.yml, or by hand:
  hf jobs uv run tests/jobs/real_engine_job.py --flavor a10g-small --timeout 150m

Env: MODEL, MAX_MODEL_LEN, MAX_TOKENS, RESULTS_REPO (HF dataset repo that
receives the benchmark JSON and logs; empty or unset discards them).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from auto_tune.guidellm import GuideLLMError, GuideLLMRunner
from auto_tune.tuner import evaluate_slos

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "8192")
MAX_TOKENS = os.environ.get("MAX_TOKENS")
PORT = 8000
TARGET = f"http://127.0.0.1:{PORT}"
WORK = Path("/tmp/real-engine")
DATA = WORK / "data"
RESULTS = WORK / "results"
GUIDELLM = str(Path(sys.executable).parent / "guidellm")
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered"
    "/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)
SLOS = {
    "min_success_rate": 0.99,
    "max_ttft_p99_ms": 200.0,
    "max_itl_p99_ms": 50.0,
}
# Ceiling for a single benchmark, enforced inside GuideLLM as a max_duration
# constraint so a stalled run stops itself and still writes a report. The HF
# Jobs --timeout is the hard stop for a process that wedges below that level.
MAX_BENCHMARK_SECONDS = 1800

summary: dict[str, dict] = {}


def log(msg: str) -> None:
    print(f"[real-engine] {msg}", flush=True)


def vllm_log_tail(lines: int = 400) -> str:
    path = WORK / "vllm.log"
    if not path.exists():
        return "(no vllm.log)"
    return "\n".join(path.read_text().splitlines()[-lines:])


def report_gpu() -> None:
    try:
        import torch

        available = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if available else "none"
        log(f"cuda available={available} device={name} torch={torch.__version__}")
    except Exception as error:
        log(f"cuda check failed: {error}")


def start_vllm() -> subprocess.Popen:
    log(f"starting vLLM for {MODEL}")
    env = {
        **os.environ,
        # The uv base image has no CUDA toolkit; FlashInfer would JIT-compile
        # its sampler and fail on missing nvcc.
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--port", str(PORT),
            "--max-model-len", MAX_MODEL_LEN,
            "--gpu-memory-utilization", "0.85",
            "--enforce-eager",
        ],
        stdout=(WORK / "vllm.log").open("w"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup:\n{vllm_log_tail()}")
        try:
            with urllib.request.urlopen(f"{TARGET}/health", timeout=3) as resp:
                if resp.status == 200:
                    log("vLLM ready")
                    return proc
        except Exception:
            time.sleep(5)
    raise RuntimeError(f"vLLM not ready in 900 s:\n{vllm_log_tail()}")


def fetch_sharegpt_sample(n: int = 200) -> Path:
    out = DATA / "sharegpt_sample.json"
    req = urllib.request.Request(SHAREGPT_URL, headers={"Range": "bytes=0-3000000"})
    raw = urllib.request.urlopen(req, timeout=180).read().decode("utf-8", "replace")

    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(raw):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                objs.append(raw[start : i + 1])

    clean = []
    for text in objs:
        row = json.loads(text)
        turns = row.get("conversations", [])
        if len(turns) < 2 or turns[0].get("from") != "human":
            continue
        if not turns[0].get("value", "").strip():
            continue
        # The extractor joins every human turn; keep rows below max_model_len.
        human_chars = sum(len(t["value"]) for t in turns if t.get("from") == "human")
        if human_chars > 8000:
            continue
        clean.append(row)
        if len(clean) >= n:
            break

    out.write_text(json.dumps(clean))
    log(f"ShareGPT sample: {len(clean)} conversations")
    return out


def make_trace() -> tuple[Path, float]:
    rng = random.Random(42)
    rows = []
    for burst_start in (0.0, 10.0, 20.0):
        for i in range(20):
            rows.append({
                "timestamp": burst_start + i * 0.1,
                "input_length": rng.randint(100, 400),
                "output_length": rng.randint(32, 128),
            })
    path = DATA / "trace.jsonl"
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    log(f"trace: {len(rows)} rows, span {rows[-1]['timestamp']}s")
    return path, rows[-1]["timestamp"]


def check_slos(metrics: dict[str, float]) -> tuple[list[str], bool]:
    """Format the tuner's own SLO verdicts for the summary block."""
    checks, ok = evaluate_slos(SLOS, metrics)
    lines = []
    for check in checks:
        name = next(key for key in check if key.startswith(("min_", "max_")))
        threshold, value = check[name], check[name[4:]]
        lines.append(f"{'ok  ' if check['meets'] else 'MISS'} {name}={threshold} actual={value:.2f}")
    return lines, ok


def run_guidellm(
    name: str,
    *,
    profile: dict[str, Any],
    data: list[dict[str, Any] | str],
    constraints: list[dict[str, Any]] | None = None,
    arguments: dict[str, Any] | None = None,
) -> dict[str, float] | None:
    """Run one benchmark through the production runner and record its verdict."""
    backend: dict[str, Any] = {"kind": "openai_http", "target": TARGET, "model": MODEL}
    if MAX_TOKENS:
        backend["max_tokens"] = int(MAX_TOKENS)

    log(f"{name}: running")
    started = time.monotonic()
    try:
        metrics = GuideLLMRunner(command=GUIDELLM).run(
            target=TARGET,
            data=data,
            profile=profile,
            constraints=[
                *(constraints or []),
                {"kind": "max_duration", "seconds": MAX_BENCHMARK_SECONDS},
            ],
            output_path=RESULTS / f"{name}.json",
            backend=backend,
            options={
                "sample_size": 0,
                "arguments": {
                    "tokenizer": {"kind": "huggingface_auto", "model": MODEL},
                    **(arguments or {}),
                },
            },
            show_console=False,
        )
    except GuideLLMError as error:
        reason = str(error)
    else:
        elapsed = time.monotonic() - started
        slo_lines, slo_ok = check_slos(metrics)
        summary[name] = {"status": "OK", "slo_met": slo_ok, "metrics": metrics,
                         "slos": slo_lines, "elapsed_s": elapsed}
        log(f"{name}: {metrics['throughput']:.2f} req/s | "
            f"ttft p50 {metrics['ttft_p50_ms']:.0f} ms p99 {metrics['ttft_p99_ms']:.0f} ms | "
            f"itl p99 {metrics['itl_p99_ms']:.1f} ms | slo_met={slo_ok} | {elapsed:.0f} s")
        return metrics

    # GuideLLM's own console output is suppressed in this headless job, so the
    # vLLM log is the only diagnostic left when a benchmark fails.
    summary[name] = {"status": "FAIL", "reason": reason, "log_tail": vllm_log_tail(60)}
    log(f"{name}: FAIL ({reason})")
    return None


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    sharegpt_path = fetch_sharegpt_sample()
    trace_path, _span = make_trace()
    report_gpu()
    server = start_vllm()

    try:
        throughput = {"kind": "throughput", "max_concurrency": 32}
        sharegpt_data: list[dict[str, Any] | str] = [{
            "kind": "json_file",
            "path": str(sharegpt_path),
            "load_kwargs": {"split": "train"},
        }]
        sharegpt_arguments = {
            "data_column_mapper": {
                "kind": "generative_column_mapper",
                "column_mappings": {"text_column": "conversations"},
            },
            "data_preprocessor": {"kind": "tool_calling_message_extractor"},
        }

        sharegpt_metrics = run_guidellm(
            "sharegpt",
            profile=throughput,
            data=sharegpt_data,
            constraints=[{"kind": "max_requests", "count": 100}],
            arguments=sharegpt_arguments,
        )

        run_guidellm(
            "hf_dataset",
            profile=throughput,
            data=[{
                "kind": "huggingface",
                "source": "garage-bAInd/Open-Platypus",
                "load_kwargs": {"split": "train"},
            }],
            constraints=[{"kind": "max_requests", "count": 100}],
            arguments={
                "data_column_mapper": {
                    "kind": "generative_column_mapper",
                    "column_mappings": {"text_column": "instruction"},
                },
                "data_loader": {"kind": "pytorch", "samples": 200},
            },
        )

        run_guidellm(
            "trace",
            profile={"kind": "replay", "time_scale": 1.0},
            data=[{"kind": "trace_synthetic", "path": str(trace_path)}],
        )

        if sharegpt_metrics:
            rate = max(sharegpt_metrics["throughput"] * 0.5, 0.5)
            run_guidellm(
                "sharegpt_rate",
                profile={"kind": "constant", "rate": round(rate, 3), "max_concurrency": 32},
                data=sharegpt_data,
                constraints=[{"kind": "max_requests", "count": 60}],
                arguments=sharegpt_arguments,
            )
    finally:
        server.terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()

    print("\n=== REAL ENGINE SUMMARY ===")
    print(f"model={MODEL} slos={SLOS}")
    for name, result in summary.items():
        print(f"\n--- {name}: {result['status']}")
        if result["status"] != "OK":
            print(f"    {result['reason']}")
            print("    --- vLLM log tail ---")
            print(result.get("log_tail", ""))
            continue
        m = result["metrics"]
        print(f"    throughput      {m['throughput']:.2f} req/s "
              f"({m['output_tokens_per_second']:.0f} out tok/s, "
              f"{m['total_tokens_per_second']:.0f} total tok/s)")
        print(f"    ttft            p50 {m['ttft_p50_ms']:.1f} ms | p99 {m['ttft_p99_ms']:.1f} ms")
        print(f"    itl p99         {m['itl_p99_ms']:.2f} ms")
        print(f"    e2e p99         {m['e2e_p99_ms']:.0f} ms")
        print(f"    requests        {m['successful_requests']:.0f} ok / "
              f"{m['total_requests']:.0f} ({m['success_rate']:.1%}) "
              f"in {result['elapsed_s']:.1f} s")
        for line in result["slos"]:
            print(f"    slo  {line}")

    results_repo = os.environ.get("RESULTS_REPO")
    if results_repo and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            folder_path=str(RESULTS),
            repo_id=results_repo,
            repo_type="dataset",
            token=os.environ["HF_TOKEN"],
        )
        log(f"results uploaded to https://huggingface.co/datasets/{results_repo}")

    failed = [name for name, result in summary.items() if result["status"] != "OK"]
    print(f"\nVERDICT: {'benchmark failures: ' + ', '.join(failed) if failed else 'all benchmarks ran'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
