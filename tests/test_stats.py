"""Tests for perftok.stats — percentile math and report computation."""

from __future__ import annotations

import pytest

from perftok.models import RequestResult
from perftok.stats import compute_latency_stats, compute_report


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
                inter_chunk_latencies_ms=[10.0, 12.0],
                inter_token_latency_ms=7.5,
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

    def test_itl_icl_and_per_user_throughput(self):
        results = [
            RequestResult(
                success=True, ttft_ms=50.0, e2e_latency_ms=250.0, output_tokens=21,
                inter_chunk_latencies_ms=[10.0, 10.0], inter_token_latency_ms=10.0,
            ),
            RequestResult(
                success=True, ttft_ms=50.0, e2e_latency_ms=450.0, output_tokens=21,
                inter_chunk_latencies_ms=[20.0, 20.0, 20.0], inter_token_latency_ms=20.0,
            ),
            RequestResult(success=True, ttft_ms=50.0, e2e_latency_ms=50.0, output_tokens=1),
        ]
        report = compute_report(results, total_duration_s=1.0)

        assert report.itl_stats.min == 10.0  # per request, from inter_token_latency_ms
        assert report.itl_stats.max == 20.0
        assert report.icl_stats.min == 10.0  # pooled chunk gaps
        assert report.icl_stats.max == 20.0
        assert report.output_throughput_per_user_stats.min == 50.0  # 1000 / 20
        assert report.output_throughput_per_user_stats.max == 100.0  # 1000 / 10

    def test_output_length_mismatch(self):
        """Mismatch when |actual - requested| > min(5% of requested, 50) tokens."""

        def r(actual: int, requested: int) -> RequestResult:
            return RequestResult(
                success=True, e2e_latency_ms=100.0, output_tokens=actual,
                requested_output_tokens=requested,
            )

        results = [r(100, 100), r(98, 100), r(50, 100), r(2000, 2000), r(1940, 2000)]
        report = compute_report(results, total_duration_s=1.0)

        assert report.output_tokens_mean == pytest.approx((100 + 98 + 50 + 2000 + 1940) / 5)
        assert report.requested_output_tokens_mean == pytest.approx(4300 / 5)
        assert report.output_length_mismatch_count == 2  # 50/100 and 1940/2000
        assert report.output_length_mismatch_rate == pytest.approx(40.0)

    def test_no_requested_tokens_means_no_mismatch(self):
        results = [RequestResult(success=True, e2e_latency_ms=1.0, output_tokens=5)]
        report = compute_report(results, total_duration_s=1.0)

        assert report.requested_output_tokens_mean is None
        assert report.output_length_mismatch_count == 0

    def test_mixed_success_failure(self):
        results = [
            RequestResult(
                success=True,
                ttft_ms=50.0,
                e2e_latency_ms=200.0,
                output_tokens=20,
                inter_chunk_latencies_ms=[10.0],
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
