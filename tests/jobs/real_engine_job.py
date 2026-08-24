# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "vllm",
#   "guidellm[recommended]==0.7.3",
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
  hf jobs uv run tests/jobs/real_engine_job.py --flavor a10g-small --timeout 45m

Env: MODEL, MAX_MODEL_LEN, MAX_TOKENS.
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


def metrics_from(report_path: Path) -> dict[str, float]:
    from guidellm.benchmark import GenerativeBenchmarksReport

    parsed = GenerativeBenchmarksReport.load_file(str(report_path))
    benchmark = max(
        parsed.benchmarks,
        key=lambda b: b.metrics.requests_per_second.successful.mean,
    )
    m = benchmark.metrics
    totals = m.request_totals
    return {
        "throughput": m.requests_per_second.successful.mean,
        "success_rate": (totals.successful / totals.total) if totals.total else 0.0,
        "successful": float(totals.successful),
        "duration_s": benchmark.duration,
        "output_tokens_per_second": m.output_tokens_per_second.successful.mean,
        "prompt_tokens_mean": m.prompt_token_count.successful.mean,
        "output_tokens_mean": m.output_token_count.successful.mean,
        "ttft_p50_ms": m.time_to_first_token_ms.successful.percentiles.p50,
        "ttft_p99_ms": m.time_to_first_token_ms.successful.percentiles.p99,
        "itl_p99_ms": m.inter_token_latency_ms.successful.percentiles.p99,
        "e2e_p99_ms": m.request_latency.successful.percentiles.p99 * 1000,
    }


def check_slos(metrics: dict[str, float]) -> tuple[list[str], bool]:
    results, ok = [], True
    for name, threshold in SLOS.items():
        metric = name.removeprefix("min_").removeprefix("max_")
        value = metrics[metric]
        passed = value >= threshold if name.startswith("min_") else value <= threshold
        ok = ok and passed
        results.append(f"{'ok  ' if passed else 'MISS'} {name}={threshold} actual={value:.2f}")
    return results, ok


def run_guidellm(name: str, extra_args: list[str]) -> dict[str, float] | None:
    report = RESULTS / f"{name}.json"
    cmd = [
        GUIDELLM, "run",
        "--backend", json.dumps(
            {"kind": "openai_http", "target": TARGET, "model": MODEL}
            | ({"max_tokens": int(MAX_TOKENS)} if MAX_TOKENS else {})
        ),
        "--tokenizer", f"kind=huggingface_auto,model={MODEL}",
        "--output", f"kind=json,path={report}",
        "--metrics", "kind=generative,sample_size=0",
        "--disable-console-interactive",
        *extra_args,
    ]
    log(f"{name}: running")
    log_path = RESULTS / f"{name}.log"
    with log_path.open("w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=1800)
    if proc.returncode != 0:
        summary[name] = {"status": "FAIL", "reason": f"guidellm exit {proc.returncode}",
                         "log_tail": log_path.read_text()[-1500:]}
        log(f"{name}: FAIL (exit {proc.returncode})")
        return None

    metrics = metrics_from(report)
    slo_lines, slo_ok = check_slos(metrics)
    summary[name] = {"status": "OK", "slo_met": slo_ok, "metrics": metrics, "slos": slo_lines}
    log(f"{name}: {metrics['throughput']:.2f} req/s | "
        f"ttft p50 {metrics['ttft_p50_ms']:.0f} ms p99 {metrics['ttft_p99_ms']:.0f} ms | "
        f"itl p99 {metrics['itl_p99_ms']:.1f} ms | slo_met={slo_ok}")
    return metrics


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    sharegpt_path = fetch_sharegpt_sample()
    trace_path, _span = make_trace()
    report_gpu()
    server = start_vllm()

    try:
        sharegpt_metrics = run_guidellm("sharegpt", [
            "--profile", "kind=throughput,max_concurrency=32",
            "--constraint", "kind=max_requests,count=100",
            "--data", json.dumps({
                "kind": "json_file",
                "path": str(sharegpt_path),
                "load_kwargs": {"split": "train"},
            }),
            "--data-column-mapper", json.dumps({
                "kind": "generative_column_mapper",
                "column_mappings": {"text_column": "conversations"},
            }),
            "--data-preprocessor", "kind=tool_calling_message_extractor",
        ])

        run_guidellm("hf_dataset", [
            "--profile", "kind=throughput,max_concurrency=32",
            "--constraint", "kind=max_requests,count=100",
            "--data", json.dumps({
                "kind": "huggingface",
                "source": "garage-bAInd/Open-Platypus",
                "load_kwargs": {"split": "train"},
            }),
            "--data-column-mapper", json.dumps({
                "kind": "generative_column_mapper",
                "column_mappings": {"text_column": "instruction"},
            }),
            "--data-loader", "kind=pytorch,samples=200",
        ])

        run_guidellm("trace", [
            "--profile", "kind=replay,time_scale=1.0",
            "--data", f"kind=trace_synthetic,path={trace_path}",
        ])

        if sharegpt_metrics:
            rate = max(sharegpt_metrics["throughput"] * 0.5, 0.5)
            run_guidellm("sharegpt_rate", [
                "--profile", f"kind=constant,rate={rate:.3f},max_concurrency=32",
                "--constraint", "kind=max_requests,count=60",
                "--data", json.dumps({
                    "kind": "json_file",
                    "path": str(sharegpt_path),
                    "load_kwargs": {"split": "train"},
                }),
                "--data-column-mapper", json.dumps({
                    "kind": "generative_column_mapper",
                    "column_mappings": {"text_column": "conversations"},
                }),
                "--data-preprocessor", "kind=tool_calling_message_extractor",
            ])
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
            print(result.get("log_tail", ""))
            continue
        m = result["metrics"]
        print(f"    throughput      {m['throughput']:.2f} req/s "
              f"({m['output_tokens_per_second']:.0f} out tok/s)")
        print(f"    tokens          prompt {m['prompt_tokens_mean']:.0f} / "
              f"output {m['output_tokens_mean']:.0f}")
        print(f"    ttft            p50 {m['ttft_p50_ms']:.1f} ms | p99 {m['ttft_p99_ms']:.1f} ms")
        print(f"    itl p99         {m['itl_p99_ms']:.2f} ms")
        print(f"    e2e p99         {m['e2e_p99_ms']:.0f} ms")
        print(f"    requests        {m['successful']:.0f} in {m['duration_s']:.1f} s")
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
