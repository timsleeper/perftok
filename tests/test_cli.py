"""Tests for tokenflow.cli — Click CLI integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from tokenflow.cli import main
from tokenflow.models import BenchmarkReport, LatencyStats


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def sample_report():
    stats = LatencyStats(
        mean=10.0, stddev=2.0, p50=9.5, p75=11.0,
        p90=13.0, p95=14.0, p99=15.0, min=5.0, max=18.0,
    )
    return BenchmarkReport(
        total_requests=10,
        successful_requests=10,
        failed_requests=0,
        total_duration_s=1.0,
        ttft_stats=stats,
        itl_stats=stats,
        e2e_latency_stats=stats,
        output_throughput_per_request_stats=stats,
        output_token_throughput=100.0,
        request_throughput=10.0,
        error_rate=0.0,
    )


class TestMainGroup:
    def test_help_exits_zero(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Lightweight benchmarking" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestRunCommand:
    def test_run_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--url" in result.output
        assert "--concurrency" in result.output
        assert "--streaming" in result.output
        assert "--output-format" in result.output

    def test_missing_required_args(self, runner):
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_run_with_json_output(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("tokenflow.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "run",
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--num-requests", "10",
                    "--output-format", "json",
                ],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_requests"] == 10

    def test_run_with_table_output(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("tokenflow.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "run",
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--output-format", "table",
                ],
            )
        assert result.exit_code == 0
        assert "TTFT" in result.output

    def test_run_with_file_output(self, runner, sample_report, tmp_path):
        outfile = str(tmp_path / "out.json")
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("tokenflow.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "run",
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--output-format", "json",
                    "--output-file", outfile,
                ],
            )
        assert result.exit_code == 0
        with open(outfile) as f:
            data = json.loads(f.read())
        assert data["total_requests"] == 10

    def test_run_non_streaming_flag(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("tokenflow.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "run",
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--no-streaming",
                    "--output-format", "json",
                ],
            )
        assert result.exit_code == 0
        # Verify the config passed to run_benchmark had streaming=False
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.streaming is False
