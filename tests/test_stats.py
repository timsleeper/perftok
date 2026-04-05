"""Tests for tokenflow.stats — percentile math and report computation."""

from __future__ import annotations

import pytest

from tokenflow.models import RequestResult
from tokenflow.stats import compute_latency_stats, compute_report


class TestComputeLatencyStats:
    def test_known_values(self):
        values = list(range(1, 101))  # 1..100
        stats = compute_latency_stats(values)
        assert stats.mean == pytest.approx(50.5)
        assert stats.min == 1.0
        assert stats.max == 100.0
        assert stats.p50 == pytest.approx(50.5)
        assert stats.p99 == pytest.approx(99.01, abs=0.5)

    def test_single_value(self):
        stats = compute_latency_stats([42.0])
        assert stats.mean == 42.0
        assert stats.stddev == 0.0
        assert stats.p50 == 42.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_empty_returns_none(self):
        assert compute_latency_stats([]) is None

    def test_two_values(self):
        stats = compute_latency_stats([10.0, 20.0])
        assert stats.mean == 15.0
        assert stats.min == 10.0
        assert stats.max == 20.0


class TestComputeReport:
    def test_all_successful(self):
        results = [
            RequestResult(
                success=True,
                ttft_ms=50.0 + i,
                e2e_latency_ms=200.0 + i,
                output_tokens=20,
                inter_token_latencies_ms=[10.0, 12.0],
            )
            for i in range(10)
        ]
        report = compute_report(results, total_duration_s=2.0)
        assert report.total_requests == 10
        assert report.successful_requests == 10
        assert report.failed_requests == 0
        assert report.error_rate == 0.0
        assert report.request_throughput == pytest.approx(5.0)
        assert report.ttft_stats is not None
        assert report.itl_stats is not None
        assert report.e2e_latency_stats is not None
        assert report.output_token_throughput > 0

    def test_mixed_success_failure(self):
        results = [
            RequestResult(
                success=True,
                ttft_ms=50.0,
                e2e_latency_ms=200.0,
                output_tokens=20,
                inter_token_latencies_ms=[10.0],
            ),
            RequestResult(success=False, e2e_latency_ms=100.0, error="timeout"),
            RequestResult(success=False, e2e_latency_ms=50.0, error="500"),
        ]
        report = compute_report(results, total_duration_s=1.0)
        assert report.total_requests == 3
        assert report.successful_requests == 1
        assert report.failed_requests == 2
        assert report.error_rate == pytest.approx(200.0 / 3.0, abs=0.1)

    def test_all_failures(self):
        results = [
            RequestResult(success=False, e2e_latency_ms=100.0, error="err")
            for _ in range(5)
        ]
        report = compute_report(results, total_duration_s=1.0)
        assert report.total_requests == 5
        assert report.successful_requests == 0
        assert report.error_rate == 100.0
        assert report.ttft_stats is None
        assert report.itl_stats is None

    def test_empty_results(self):
        report = compute_report([], total_duration_s=1.0)
        assert report.total_requests == 0
        assert report.error_rate == 0.0
        assert report.output_token_throughput == 0.0
