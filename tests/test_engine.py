"""Tests for tokenflow.engine — asyncio benchmark orchestrator."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from tokenflow.engine import run_benchmark
from tokenflow.models import BenchmarkConfig, RequestResult


def _make_config(**overrides) -> BenchmarkConfig:
    defaults = dict(model="test", url="http://localhost:8000", concurrency=2, num_requests=5)
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _fake_result(i: int) -> RequestResult:
    return RequestResult(
        success=True,
        ttft_ms=10.0 + i,
        e2e_latency_ms=100.0 + i,
        output_tokens=10,
        inter_token_latencies_ms=[5.0],
    )


class TestRunBenchmark:
    @pytest.mark.asyncio
    async def test_request_count(self):
        """Engine sends exactly num_requests requests."""
        call_count = 0

        async def mock_send(session, config, prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            return _fake_result(call_count)

        config = _make_config(num_requests=10)
        with patch("tokenflow.engine.send_request", side_effect=mock_send):
            with patch("tokenflow.engine.generate_prompt", return_value="test"):
                with patch(
                    "tokenflow.engine.generate_output_token_count", return_value=50
                ):
                    report = await run_benchmark(config)

        assert call_count == 10
        assert report.total_requests == 10
        assert report.successful_requests == 10

    @pytest.mark.asyncio
    async def test_concurrency_limit(self):
        """Engine never exceeds the configured concurrency."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_send(session, config, prompt, max_tokens):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.01)
            async with lock:
                current_concurrent -= 1
            return _fake_result(0)

        config = _make_config(concurrency=3, num_requests=20)
        with patch("tokenflow.engine.send_request", side_effect=mock_send):
            with patch("tokenflow.engine.generate_prompt", return_value="test"):
                with patch(
                    "tokenflow.engine.generate_output_token_count", return_value=50
                ):
                    await run_benchmark(config)

        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_partial_failures(self):
        """Engine handles a mix of successful and failed requests."""
        call_count = 0

        async def mock_send(session, config, prompt, max_tokens):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                return RequestResult(
                    success=False, e2e_latency_ms=50.0, error="mock error"
                )
            return _fake_result(call_count)

        config = _make_config(num_requests=9)
        with patch("tokenflow.engine.send_request", side_effect=mock_send):
            with patch("tokenflow.engine.generate_prompt", return_value="test"):
                with patch(
                    "tokenflow.engine.generate_output_token_count", return_value=50
                ):
                    report = await run_benchmark(config)

        assert report.total_requests == 9
        assert report.failed_requests == 3
        assert report.successful_requests == 6

    @pytest.mark.asyncio
    async def test_report_has_duration(self):
        """Report includes a positive total duration."""

        async def mock_send(session, config, prompt, max_tokens):
            return _fake_result(0)

        config = _make_config(num_requests=3)
        with patch("tokenflow.engine.send_request", side_effect=mock_send):
            with patch("tokenflow.engine.generate_prompt", return_value="test"):
                with patch(
                    "tokenflow.engine.generate_output_token_count", return_value=50
                ):
                    report = await run_benchmark(config)

        assert report.total_duration_s > 0
