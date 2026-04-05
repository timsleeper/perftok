"""Click CLI with benchmark command."""

from __future__ import annotations

import asyncio

import click
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from perftok._version import __version__
from perftok.client import fetch_models
from perftok.engine import run_benchmark
from perftok.formatter import write_output
from perftok.models import BenchmarkConfig

_HELP_EPILOG = """
Example:

  perftok --url http://localhost:8000 --num-requests 100 --concurrency 10
"""


@click.command(epilog=_HELP_EPILOG)
@click.version_option(version=__version__, prog_name="perftok")
@click.option(
    "--model", default=None,
    help="Model name (auto-discovered if omitted).",
)
@click.option(
    "--url", required=True,
    help="Base URL of the endpoint.",
)
@click.option(
    "--api-key", default=None, envvar="API_KEY",
    help="API key (or set API_KEY env var).",
)
@click.option(
    "--concurrency", default=1, type=int, show_default=True,
    help="Max concurrent requests.",
)
@click.option(
    "--num-requests", default=1, type=int, show_default=True,
    help="Total number of requests.",
)
@click.option(
    "--mean-input-tokens", default=550, type=int, show_default=True,
    help="Mean input prompt tokens.",
)
@click.option(
    "--stddev-input-tokens", default=150, type=int, show_default=True,
    help="Stddev of input tokens.",
)
@click.option(
    "--mean-output-tokens", default=150, type=int, show_default=True,
    help="Mean output tokens.",
)
@click.option(
    "--stddev-output-tokens", default=10, type=int, show_default=True,
    help="Stddev of output tokens.",
)
@click.option(
    "--timeout", default=300, type=int, show_default=True,
    help="Request timeout in seconds.",
)
@click.option(
    "--streaming/--no-streaming", default=True, show_default=True,
    help="Use streaming SSE.",
)
@click.option(
    "--insecure", is_flag=True, default=False,
    help="Skip TLS/SSL certificate verification.",
)
@click.option(
    "--output-format",
    type=click.Choice(["table", "json", "csv"]),
    default="table", show_default=True,
    help="Output format.",
)
@click.option(
    "--output-file", default=None, type=str,
    help="Write results to file.",
)
def main(
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
    insecure: bool,
    output_format: str,
    output_file: str | None,
) -> None:
    """Lightweight benchmarking tool for OpenAI-compatible LLM endpoints."""
    if model is None:
        model = _discover_model(url, api_key, insecure)

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
        insecure=insecure,
    )

    progress = Progress(
        TextColumn("[bold blue]Benchmarking"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed}/{task.total} requests"),
        transient=True,
    )

    with progress:
        task_id = progress.add_task("requests", total=num_requests)

        def on_progress(completed: int, total: int) -> None:
            progress.update(task_id, completed=completed)

        report = asyncio.run(run_benchmark(config, on_progress=on_progress))

    output = write_output(
        report, format_name=output_format, output_file=output_file, config=config
    )
    click.echo(output)


def _discover_model(url: str, api_key: str | None, insecure: bool) -> str:
    """Fetch models from the endpoint and let the user pick one."""
    url = url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]

    click.echo(f"No --model specified. Fetching models from {url}...")
    try:
        models = asyncio.run(
            fetch_models(url, api_key=api_key, insecure=insecure)
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    if len(models) == 1:
        if click.confirm(f"Found model: {models[0]}. Use it?", default=True):
            return models[0]
        raise click.Abort()

    click.echo("Available models:")
    for i, m in enumerate(models, 1):
        click.echo(f"  {i}. {m}")
    choice = click.prompt(
        "Select a model", type=click.IntRange(1, len(models))
    )
    return models[choice - 1]
