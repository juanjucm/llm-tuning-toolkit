from typing import Any, Dict


def extract_inference_benchmarker_metrics(results_dict: Dict[str, Any]) -> Dict[str, Any]:
    results = results_dict.get("results", [])
    result = results[-1] if results else {}

    successful_requests = result.get("successful_requests", 0)
    total_requests = max(result.get("total_requests", 1), 1)
    ttft = result.get("time_to_first_token_ms", {})
    e2e = result.get("e2e_latency_ms", {})
    itl = result.get("inter_token_latency_ms", {})

    return {
        "throughput": result.get("request_rate", 0),
        "total_requests": result.get("total_requests", 0),
        "successful_requests": successful_requests,
        "failed_requests": result.get("failed_requests", 0),
        "success_rate": successful_requests / total_requests,
        "ttft_p99_ms": ttft.get("p99", float("inf")),
        "ttft_p95_ms": ttft.get("p95", float("inf")),
        "ttft_p90_ms": ttft.get("p90", float("inf")),
        "ttft_p80_ms": ttft.get("p80", float("inf")),
        "ttft_p70_ms": ttft.get("p70", float("inf")),
        "ttft_p60_ms": ttft.get("p60", float("inf")),
        "ttft_p50_ms": ttft.get("p50", float("inf")),
        "ttft_avg_ms": ttft.get("avg", float("inf")),
        "e2e_p99_ms": e2e.get("p99", float("inf")),
        "e2e_p95_ms": e2e.get("p95", float("inf")),
        "e2e_p90_ms": e2e.get("p90", float("inf")),
        "e2e_p80_ms": e2e.get("p80", float("inf")),
        "e2e_p70_ms": e2e.get("p70", float("inf")),
        "e2e_p60_ms": e2e.get("p60", float("inf")),
        "e2e_p50_ms": e2e.get("p50", float("inf")),
        "e2e_avg_ms": e2e.get("avg", float("inf")),
        "itl_p99_ms": itl.get("p99", float("inf")),
        "itl_p95_ms": itl.get("p95", float("inf")),
        "itl_p90_ms": itl.get("p90", float("inf")),
        "itl_p80_ms": itl.get("p80", float("inf")),
        "itl_p70_ms": itl.get("p70", float("inf")),
        "itl_p60_ms": itl.get("p60", float("inf")),
        "itl_p50_ms": itl.get("p50", float("inf")),
        "itl_avg_ms": itl.get("avg", float("inf")),
    }
