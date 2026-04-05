"""Rich table, JSON, and CSV output formatting."""

from __future__ import annotations

import csv
import io
from pathlib import Path

from rich.console import Console
from rich.table import Table

from perftok.models import BenchmarkConfig, BenchmarkReport

_STAT_FIELDS = ["mean", "stddev", "p50", "p75", "p90", "p95", "p99", "min", "max"]


def format_config_table(config: BenchmarkConfig) -> str:
    """Render benchmark configuration as a Rich table string."""
    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    tbl = Table(title="Configuration", show_header=True)
    tbl.add_column("Parameter", style="bold")
    tbl.add_column("Value", justify="right")
    tbl.add_row("Model", config.model)
    tbl.add_row("URL", config.url)
    tbl.add_row("Concurrency", str(config.concurrency))
    tbl.add_row("Num Requests", str(config.num_requests))
    tbl.add_row("Mean Input Tokens", str(config.mean_input_tokens))
    tbl.add_row("Stddev Input Tokens", str(config.stddev_input_tokens))
    tbl.add_row("Mean Output Tokens", str(config.mean_output_tokens))
    tbl.add_row("Stddev Output Tokens", str(config.stddev_output_tokens))
    tbl.add_row("Timeout (s)", str(config.timeout))
    tbl.add_row("Streaming", str(config.streaming))
    tbl.add_row("Insecure", str(config.insecure))
    console.print(tbl)

    return console.file.getvalue()


def format_json(report: BenchmarkReport) -> str:
    """Serialize report to indented JSON."""
    return report.model_dump_json(indent=2)


def format_csv(report: BenchmarkReport) -> str:
    """Serialize report to a single-row CSV with flattened stat columns."""
    flat = _flatten_report(report)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)
    return output.getvalue()


def format_table(report: BenchmarkReport) -> str:
    """Render report as a Rich table string."""
    console = Console(file=io.StringIO(), force_terminal=False, width=120)

    # Summary table
    summary = Table(title="Benchmark Summary", show_header=True)
    summary.add_column("Metric", style="bold")
    summary.add_column("Value", justify="right")
    summary.add_row("Total Requests", str(report.total_requests))
    summary.add_row("Successful", str(report.successful_requests))
    summary.add_row("Failed", str(report.failed_requests))
    summary.add_row("Duration (s)", f"{report.total_duration_s:.2f}")
    summary.add_row("Output Throughput (tok/s)", f"{report.output_token_throughput:.2f}")
    summary.add_row("Request Throughput (req/s)", f"{report.request_throughput:.2f}")
    summary.add_row("Error Rate (%)", f"{report.error_rate:.1f}")
    console.print(summary)

    # Latency breakdown table
    if any(
        s is not None
        for s in [report.ttft_stats, report.itl_stats, report.e2e_latency_stats]
    ):
        lat = Table(title="Latency Statistics (ms)", show_header=True)
        lat.add_column("Metric", style="bold")
        for f in _STAT_FIELDS:
            lat.add_column(f, justify="right")

        for name, stats in [
            ("TTFT", report.ttft_stats),
            ("ITL", report.itl_stats),
            ("E2E Latency", report.e2e_latency_stats),
            ("Output Throughput/req", report.output_throughput_per_request_stats),
        ]:
            if stats:
                lat.add_row(name, *[f"{getattr(stats, f):.2f}" for f in _STAT_FIELDS])

        console.print(lat)

    return console.file.getvalue()


def write_output(
    report: BenchmarkReport,
    format_name: str = "table",
    output_file: str | None = None,
    config: BenchmarkConfig | None = None,
) -> str:
    """Format report and optionally write to file. Returns formatted string."""
    formatters = {
        "json": format_json,
        "csv": format_csv,
        "table": format_table,
    }
    formatter = formatters[format_name]
    output = formatter(report)

    # Prepend config table for table output
    if format_name == "table" and config:
        output = format_config_table(config) + output

    if output_file:
        path = Path(output_file).resolve()
        if not path.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {path.parent}"
            )
        with open(path, "w") as f:
            f.write(output)

    return output


def _flatten_report(report: BenchmarkReport) -> dict:
    """Flatten report into a single dict for CSV output."""
    flat: dict = {
        "total_requests": report.total_requests,
        "successful_requests": report.successful_requests,
        "failed_requests": report.failed_requests,
        "total_duration_s": report.total_duration_s,
        "output_token_throughput": report.output_token_throughput,
        "request_throughput": report.request_throughput,
        "error_rate": report.error_rate,
    }

    for prefix, stats in [
        ("ttft", report.ttft_stats),
        ("itl", report.itl_stats),
        ("e2e_latency", report.e2e_latency_stats),
        ("output_throughput_per_req", report.output_throughput_per_request_stats),
    ]:
        if stats:
            for field in _STAT_FIELDS:
                flat[f"{prefix}_{field}"] = getattr(stats, field)
        else:
            for field in _STAT_FIELDS:
                flat[f"{prefix}_{field}"] = ""

    return flat
