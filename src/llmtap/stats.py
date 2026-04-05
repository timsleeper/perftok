"""Percentile/aggregation math and report computation."""

from __future__ import annotations

import math
import statistics

from llmtap.models import BenchmarkReport, LatencyStats, RequestResult


def compute_latency_stats(values: list[float]) -> LatencyStats | None:
    """Compute percentile statistics from a list of values. Returns None if empty."""
    if not values:
        return None
    n = len(values)
    sorted_vals = sorted(values)
    mean = statistics.mean(sorted_vals)
    stddev = statistics.pstdev(sorted_vals) if n > 1 else 0.0

    def percentile(pct: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    return LatencyStats(
        mean=mean,
        stddev=stddev,
        p50=percentile(50),
        p75=percentile(75),
        p90=percentile(90),
        p95=percentile(95),
        p99=percentile(99),
        min=sorted_vals[0],
        max=sorted_vals[-1],
    )


def compute_report(
    results: list[RequestResult],
    total_duration_s: float,
) -> BenchmarkReport:
    """Aggregate individual request results into a benchmark report."""
    total = len(results)
    if total == 0:
        return BenchmarkReport(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_duration_s=total_duration_s,
        )

    successful = [r for r in results if r.success]
    n_success = len(successful)
    n_fail = total - n_success
    error_rate = (n_fail / total) * 100.0

    ttft_values = [r.ttft_ms for r in successful if r.ttft_ms is not None]
    itl_values = [v for r in successful for v in r.inter_token_latencies_ms]
    e2e_values = [r.e2e_latency_ms for r in successful]
    throughput_values = [r.output_token_throughput for r in successful]

    total_output_tokens = sum(r.output_tokens for r in successful)
    output_token_throughput = (
        total_output_tokens / total_duration_s if total_duration_s > 0 else 0.0
    )
    request_throughput = total / total_duration_s if total_duration_s > 0 else 0.0

    return BenchmarkReport(
        total_requests=total,
        successful_requests=n_success,
        failed_requests=n_fail,
        total_duration_s=total_duration_s,
        ttft_stats=compute_latency_stats(ttft_values),
        itl_stats=compute_latency_stats(itl_values),
        e2e_latency_stats=compute_latency_stats(e2e_values),
        output_throughput_per_request_stats=compute_latency_stats(throughput_values),
        output_token_throughput=output_token_throughput,
        request_throughput=request_throughput,
        error_rate=error_rate,
    )
