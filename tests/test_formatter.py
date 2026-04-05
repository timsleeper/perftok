"""Tests for perftok.formatter — JSON, CSV, and Rich table output."""

from __future__ import annotations

import csv
import io
import json

import pytest

from perftok.formatter import (
    format_config_table,
    format_csv,
    format_json,
    format_table,
    write_output,
)
from perftok.models import BenchmarkConfig, BenchmarkReport, LatencyStats


@pytest.fixture
def sample_stats():
    return LatencyStats(
        mean=10.0, stddev=2.0, p50=9.5, p75=11.0,
        p90=13.0, p95=14.0, p99=15.0, min=5.0, max=18.0,
    )


@pytest.fixture
def sample_report(sample_stats):
    return BenchmarkReport(
        total_requests=100,
        successful_requests=95,
        failed_requests=5,
        total_duration_s=10.0,
        ttft_stats=sample_stats,
        itl_stats=sample_stats,
        e2e_latency_stats=sample_stats,
        output_throughput_per_request_stats=sample_stats,
        output_token_throughput=500.0,
        request_throughput=10.0,
        error_rate=5.0,
    )


class TestFormatJson:
    def test_valid_json(self, sample_report):
        output = format_json(sample_report)
        data = json.loads(output)
        assert data["total_requests"] == 100
        assert data["error_rate"] == 5.0

    def test_contains_stats(self, sample_report):
        data = json.loads(format_json(sample_report))
        assert "ttft_stats" in data
        assert data["ttft_stats"]["p50"] == 9.5


class TestFormatCsv:
    def test_headers_present(self, sample_report):
        output = format_csv(sample_report)
        reader = csv.reader(io.StringIO(output))
        headers = next(reader)
        assert "total_requests" in headers
        assert "error_rate" in headers
        assert "ttft_mean" in headers

    def test_single_data_row(self, sample_report):
        output = format_csv(sample_report)
        reader = csv.reader(io.StringIO(output))
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row

    def test_values_correct(self, sample_report):
        output = format_csv(sample_report)
        reader = csv.DictReader(io.StringIO(output))
        row = next(reader)
        assert row["total_requests"] == "100"
        assert row["error_rate"] == "5.0"


class TestFormatTable:
    def test_contains_metric_names(self, sample_report):
        output = format_table(sample_report)
        assert "TTFT" in output
        assert "ITL" in output
        assert "E2E Latency" in output
        assert "Output Throughput" in output
        assert "Error Rate" in output

    def test_contains_values(self, sample_report):
        output = format_table(sample_report)
        assert "100" in output  # total requests
        assert "5.0" in output  # error rate


class TestConfigTable:
    def test_contains_parameters(self):
        config = BenchmarkConfig(
            model="llama3", url="http://localhost:8000", concurrency=10,
            num_requests=100, streaming=True,
        )
        output = format_config_table(config)
        assert "llama3" in output
        assert "localhost" in output
        assert "10" in output
        assert "100" in output
        assert "Configuration" in output


class TestWriteOutput:
    def test_write_to_file(self, sample_report, tmp_path):
        outfile = tmp_path / "results.json"
        write_output(sample_report, format_name="json", output_file=str(outfile))
        content = outfile.read_text()
        data = json.loads(content)
        assert data["total_requests"] == 100

    def test_write_csv_to_file(self, sample_report, tmp_path):
        outfile = tmp_path / "results.csv"
        write_output(sample_report, format_name="csv", output_file=str(outfile))
        content = outfile.read_text()
        assert "total_requests" in content

    def test_nonexistent_parent_dir_rejected(self, sample_report):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            write_output(
                sample_report,
                format_name="json",
                output_file="/nonexistent/dir/results.json",
            )

    def test_path_resolved_and_written(self, sample_report, tmp_path):
        """Paths with .. are resolved before writing."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        # ../results.json from subdir resolves to tmp_path/results.json
        outfile = str(subdir / ".." / "results.json")
        write_output(sample_report, format_name="json", output_file=outfile)
        resolved = tmp_path / "results.json"
        assert resolved.exists()
