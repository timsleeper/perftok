"""Tests for perftok.cli — Click CLI integration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from perftok.cli import main
from perftok.models import BenchmarkReport, LatencyStats


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


class TestCli:
    def test_help_shows_all_options(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Lightweight benchmarking" in result.output
        assert "--model" in result.output
        assert "--url" in result.output
        assert "--concurrency" in result.output
        assert "--streaming" in result.output
        assert "--output-format" in result.output
        assert "--insecure" in result.output

    def test_help_shows_example(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "Example:" in result.output
        assert "--url" in result.output
        assert "--num-requests" in result.output

    def test_help_shows_defaults(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "[required]" in result.output
        assert "[default: 1]" in result.output or "default:" in result.output

    def test_version(self, runner):
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.2" in result.output

    def test_missing_url_exits_nonzero(self, runner):
        result = runner.invoke(main, [])
        assert result.exit_code != 0

    def test_json_output(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("perftok.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--num-requests", "10",
                    "--output-format", "json",
                ],
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_requests"] == 10

    def test_table_output_includes_config(self, runner, sample_report):
        """Table output includes a configuration summary table."""
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("perftok.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--concurrency", "5",
                    "--output-format", "table",
                ],
            )
        assert result.exit_code == 0
        assert "Configuration" in result.output
        assert "test-model" in result.output
        assert "localhost" in result.output

    def test_file_output(self, runner, sample_report, tmp_path):
        outfile = str(tmp_path / "out.json")
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("perftok.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
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

    def test_non_streaming_flag(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("perftok.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--no-streaming",
                    "--output-format", "json",
                ],
            )
        assert result.exit_code == 0
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.streaming is False

    def test_insecure_flag_passed_to_config(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        with patch("perftok.cli.run_benchmark", mock_benchmark):
            result = runner.invoke(
                main,
                [
                    "--model", "test-model",
                    "--url", "http://localhost:8000",
                    "--insecure",
                    "--output-format", "json",
                ],
            )
        assert result.exit_code == 0
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.insecure is True


class TestModelDiscovery:
    def test_single_model_defaults_to_yes(self, runner, sample_report):
        """Single model confirm defaults to Y — pressing Enter accepts."""
        mock_benchmark = AsyncMock(return_value=sample_report)
        mock_fetch = AsyncMock(return_value=["discovered-model"])
        with (
            patch("perftok.cli.run_benchmark", mock_benchmark),
            patch("perftok.cli.fetch_models", mock_fetch),
        ):
            result = runner.invoke(
                main,
                ["--url", "http://localhost:8000", "--output-format", "json"],
                input="\n",  # just press Enter — should accept
            )
        assert result.exit_code == 0
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.model == "discovered-model"

    def test_single_model_user_confirms(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        mock_fetch = AsyncMock(return_value=["discovered-model"])
        with (
            patch("perftok.cli.run_benchmark", mock_benchmark),
            patch("perftok.cli.fetch_models", mock_fetch),
        ):
            result = runner.invoke(
                main,
                ["--url", "http://localhost:8000", "--output-format", "json"],
                input="y\n",
            )
        assert result.exit_code == 0
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.model == "discovered-model"

    def test_single_model_user_rejects(self, runner):
        mock_fetch = AsyncMock(return_value=["discovered-model"])
        with patch("perftok.cli.fetch_models", mock_fetch):
            result = runner.invoke(
                main,
                ["--url", "http://localhost:8000", "--output-format", "json"],
                input="n\n",
            )
        assert result.exit_code != 0

    def test_multiple_models_user_picks(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        mock_fetch = AsyncMock(return_value=["model-a", "model-b", "model-c"])
        with (
            patch("perftok.cli.run_benchmark", mock_benchmark),
            patch("perftok.cli.fetch_models", mock_fetch),
        ):
            result = runner.invoke(
                main,
                ["--url", "http://localhost:8000", "--output-format", "json"],
                input="2\n",
            )
        assert result.exit_code == 0
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.model == "model-b"

    def test_fetch_failure_aborts(self, runner):
        mock_fetch = AsyncMock(side_effect=RuntimeError("connection refused"))
        with patch("perftok.cli.fetch_models", mock_fetch):
            result = runner.invoke(
                main,
                ["--url", "http://localhost:8000", "--output-format", "json"],
            )
        assert result.exit_code != 0
        assert "connection refused" in result.output

    def test_model_flag_skips_discovery(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        mock_fetch = AsyncMock(return_value=["other-model"])
        with (
            patch("perftok.cli.run_benchmark", mock_benchmark),
            patch("perftok.cli.fetch_models", mock_fetch),
        ):
            result = runner.invoke(
                main,
                [
                    "--model", "explicit-model",
                    "--url", "http://localhost:8000",
                    "--output-format", "json",
                ],
            )
        assert result.exit_code == 0
        mock_fetch.assert_not_called()
        call_config = mock_benchmark.call_args[0][0]
        assert call_config.model == "explicit-model"

    def test_insecure_passed_to_fetch_models(self, runner, sample_report):
        mock_benchmark = AsyncMock(return_value=sample_report)
        mock_fetch = AsyncMock(return_value=["discovered-model"])
        with (
            patch("perftok.cli.run_benchmark", mock_benchmark),
            patch("perftok.cli.fetch_models", mock_fetch),
        ):
            result = runner.invoke(
                main,
                [
                    "--url", "http://localhost:8000",
                    "--insecure",
                    "--output-format", "json",
                ],
                input="y\n",
            )
        assert result.exit_code == 0
        mock_fetch.assert_awaited_once()
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("insecure") is True
