"""Tests for llmtap.models — Pydantic validation and serialization."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from llmtap.models import (
    BenchmarkConfig,
    BenchmarkReport,
    LatencyStats,
    RequestResult,
)


class TestBenchmarkConfig:
    def test_minimal_valid(self):
        cfg = BenchmarkConfig(model="gpt-4", url="http://localhost:8000")
        assert cfg.model == "gpt-4"
        assert cfg.concurrency == 1
        assert cfg.num_requests == 1
        assert cfg.streaming is True
        assert cfg.insecure is False

    def test_url_trailing_slash_stripped(self):
        cfg = BenchmarkConfig(model="m", url="http://localhost:8000/")
        assert cfg.url == "http://localhost:8000"

    def test_url_v1_suffix_stripped(self):
        cfg = BenchmarkConfig(model="m", url="http://localhost:8000/v1/")
        assert cfg.url == "http://localhost:8000"

    def test_url_must_be_http_or_https(self):
        with pytest.raises(ValidationError, match="http.*https"):
            BenchmarkConfig(model="m", url="ftp://localhost:8000")

    def test_url_file_scheme_rejected(self):
        with pytest.raises(ValidationError, match="http.*https"):
            BenchmarkConfig(model="m", url="file:///etc/passwd")

    def test_url_must_have_host(self):
        with pytest.raises(ValidationError, match="host"):
            BenchmarkConfig(model="m", url="http://")

    def test_https_url_accepted(self):
        cfg = BenchmarkConfig(model="m", url="https://api.example.com")
        assert cfg.url == "https://api.example.com"

    def test_concurrency_must_be_positive(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(model="m", url="http://localhost:8000", concurrency=0)

    def test_concurrency_upper_bound(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(model="m", url="http://localhost:8000", concurrency=10001)

    def test_num_requests_must_be_positive(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(model="m", url="http://localhost:8000", num_requests=0)

    def test_num_requests_upper_bound(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                model="m", url="http://localhost:8000", num_requests=1_000_001
            )

    def test_mean_input_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(model="m", url="http://localhost:8000", mean_input_tokens=0)

    def test_mean_output_tokens_must_be_positive(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(
                model="m", url="http://localhost:8000", mean_output_tokens=0
            )

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(model="m", url="http://localhost:8000", timeout=0)

    def test_timeout_upper_bound(self):
        with pytest.raises(ValidationError):
            BenchmarkConfig(model="m", url="http://localhost:8000", timeout=3601)

    def test_all_fields(self):
        cfg = BenchmarkConfig(
            model="llama3",
            url="http://host:9000",
            api_key="sk-test",
            concurrency=10,
            num_requests=100,
            mean_input_tokens=550,
            stddev_input_tokens=150,
            mean_output_tokens=150,
            stddev_output_tokens=10,
            timeout=300,
            streaming=False,
            insecure=True,
        )
        assert cfg.concurrency == 10
        assert cfg.streaming is False
        assert cfg.api_key == "sk-test"
        assert cfg.insecure is True

    def test_serialization_roundtrip(self):
        cfg = BenchmarkConfig(model="m", url="http://localhost:8000", concurrency=5)
        data = json.loads(cfg.model_dump_json())
        cfg2 = BenchmarkConfig(**data)
        assert cfg == cfg2


class TestRequestResult:
    def test_successful_result(self):
        r = RequestResult(
            success=True,
            ttft_ms=50.0,
            e2e_latency_ms=200.0,
            output_tokens=20,
            inter_token_latencies_ms=[10.0, 12.0, 11.0],
        )
        assert r.success is True
        assert r.error is None
        assert r.output_token_throughput == 20 / 0.2

    def test_failed_result(self):
        r = RequestResult(
            success=False,
            e2e_latency_ms=100.0,
            error="timeout",
        )
        assert r.success is False
        assert r.error == "timeout"
        assert r.ttft_ms is None
        assert r.output_tokens == 0

    def test_throughput_zero_latency(self):
        r = RequestResult(success=True, e2e_latency_ms=0.0, output_tokens=10)
        assert r.output_token_throughput == 0.0

    def test_serialization_roundtrip(self):
        r = RequestResult(
            success=True, ttft_ms=10.0, e2e_latency_ms=100.0, output_tokens=5
        )
        data = json.loads(r.model_dump_json())
        r2 = RequestResult(**data)
        assert r == r2


class TestLatencyStats:
    def test_all_fields(self):
        stats = LatencyStats(
            mean=10.0,
            stddev=2.0,
            p50=9.5,
            p75=11.0,
            p90=13.0,
            p95=14.0,
            p99=15.0,
            min=5.0,
            max=18.0,
        )
        assert stats.p50 == 9.5
        assert stats.min == 5.0


class TestBenchmarkReport:
    def test_full_report(self):
        lstats = LatencyStats(
            mean=10.0, stddev=2.0, p50=9.5, p75=11.0,
            p90=13.0, p95=14.0, p99=15.0, min=5.0, max=18.0,
        )
        report = BenchmarkReport(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_duration_s=10.0,
            ttft_stats=lstats,
            itl_stats=lstats,
            e2e_latency_stats=lstats,
            output_throughput_per_request_stats=lstats,
            output_token_throughput=500.0,
            request_throughput=9.5,
            error_rate=5.0,
        )
        assert report.error_rate == 5.0
        assert report.request_throughput == 9.5

    def test_serialization_roundtrip(self):
        lstats = LatencyStats(
            mean=10.0, stddev=2.0, p50=9.5, p75=11.0,
            p90=13.0, p95=14.0, p99=15.0, min=5.0, max=18.0,
        )
        report = BenchmarkReport(
            total_requests=10,
            successful_requests=10,
            failed_requests=0,
            total_duration_s=1.0,
            ttft_stats=lstats,
            itl_stats=lstats,
            e2e_latency_stats=lstats,
            output_throughput_per_request_stats=lstats,
            output_token_throughput=100.0,
            request_throughput=10.0,
            error_rate=0.0,
        )
        data = json.loads(report.model_dump_json())
        report2 = BenchmarkReport(**data)
        assert report == report2
