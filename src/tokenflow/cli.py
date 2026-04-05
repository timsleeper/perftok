"""Click CLI group with 'run' command."""

from __future__ import annotations

import asyncio

import click

from tokenflow._version import __version__
from tokenflow.client import fetch_models
from tokenflow.engine import run_benchmark
from tokenflow.formatter import write_output
from tokenflow.models import BenchmarkConfig


@click.group()
@click.version_option(version=__version__, prog_name="tokenflow")
def main() -> None:
    """Lightweight benchmarking tool for OpenAI-compatible LLM endpoints."""


@main.command()
@click.option("--model", default=None, help="Model name (auto-discovered if omitted)")
@click.option("--url", required=True, help="Base URL of the OpenAI-compatible endpoint")
@click.option("--api-key", default=None, envvar="API_KEY", help="API key (or set API_KEY env var)")
@click.option("--concurrency", default=1, type=int, help="Max concurrent requests")
@click.option("--num-requests", default=1, type=int, help="Total number of requests")
@click.option("--mean-input-tokens", default=550, type=int, help="Mean input prompt tokens")
@click.option("--stddev-input-tokens", default=150, type=int, help="Stddev of input tokens")
@click.option("--mean-output-tokens", default=150, type=int, help="Mean output tokens")
@click.option("--stddev-output-tokens", default=10, type=int, help="Stddev of output tokens")
@click.option("--timeout", default=300, type=int, help="Request timeout in seconds")
@click.option("--streaming/--no-streaming", default=True, help="Use streaming SSE responses")
@click.option(
    "--output-format",
    type=click.Choice(["table", "json", "csv"]),
    default="table",
    help="Output format",
)
@click.option("--output-file", default=None, type=str, help="Write results to file")
def run(
    model: str | None,
    url: str,
    api_key: str | None,
    concurrency: int,
    num_requests: int,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    timeout: int,
    streaming: bool,
    output_format: str,
    output_file: str | None,
) -> None:
    """Run a benchmark against an OpenAI-compatible endpoint."""
    if model is None:
        model = _discover_model(url, api_key)

    config = BenchmarkConfig(
        model=model,
        url=url,
        api_key=api_key,
        concurrency=concurrency,
        num_requests=num_requests,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        timeout=timeout,
        streaming=streaming,
    )

    report = asyncio.run(run_benchmark(config))
    output = write_output(report, format_name=output_format, output_file=output_file)
    click.echo(output)


def _discover_model(url: str, api_key: str | None) -> str:
    """Fetch models from the endpoint and let the user pick one."""
    # Normalize URL the same way BenchmarkConfig does
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]

    click.echo(f"No --model specified. Fetching models from {url}...")
    try:
        models = asyncio.run(fetch_models(url, api_key=api_key))
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if len(models) == 1:
        if click.confirm(f"Found model: {models[0]}. Use it?"):
            return models[0]
        raise click.Abort()

    click.echo("Available models:")
    for i, m in enumerate(models, 1):
        click.echo(f"  {i}. {m}")
    choice = click.prompt(
        "Select a model", type=click.IntRange(1, len(models))
    )
    return models[choice - 1]
