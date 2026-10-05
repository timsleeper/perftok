"""asyncio benchmark orchestrator with Semaphore-based concurrency."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable

import aiohttp
import click

from perftok.client import check_ssl, send_request
from perftok.models import BenchmarkConfig, BenchmarkReport, RequestResult
from perftok.prompt import generate_prompt, sample_token_count
from perftok.stats import compute_report


async def run_benchmark(
    config: BenchmarkConfig,
    on_progress: Callable[[int, int], None] | None = None,
) -> BenchmarkReport:
    """Run the full benchmark and return an aggregated report."""
    if config.random_seed is not None:
        random.seed(config.random_seed)

    if config.insecure:
        click.echo("TLS/SSL certificate verification is disabled (--insecure).")
        ssl_param: bool | None = False  # noqa: S507
    else:
        await check_ssl(config.url, config.api_key)
        ssl_param = None

    # Default connector limit is 100, which would silently cap concurrency.
    connector = aiohttp.TCPConnector(ssl=ssl_param, limit=config.concurrency)

    # Build every prompt up front. Tokenizing is synchronous CPU work; doing it
    # inside the request tasks blocks the event loop while early responses sit
    # unread, inflating their TTFT and the total duration.
    warmup_jobs = _generate_jobs(config, config.warmup_requests)
    jobs = _generate_jobs(config, config.num_requests)

    timeout = aiohttp.ClientTimeout(total=config.timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        if warmup_jobs:
            click.echo(f"Warming up with {len(warmup_jobs)} requests...")
            await _run_jobs(session, config, warmup_jobs)

        start = time.perf_counter()
        results = await _run_jobs(session, config, jobs, on_progress)
        total_duration = time.perf_counter() - start

    return compute_report(results, total_duration_s=total_duration)


def _generate_jobs(config: BenchmarkConfig, count: int) -> list[tuple[str, int]]:
    """Sample *count* (prompt, max_tokens) pairs from the configured distributions."""
    return [
        (
            generate_prompt(
                sample_token_count(config.mean_input_tokens, config.stddev_input_tokens)
            ),
            sample_token_count(config.mean_output_tokens, config.stddev_output_tokens),
        )
        for _ in range(count)
    ]


async def _run_jobs(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    jobs: list[tuple[str, int]],
    on_progress: Callable[[int, int], None] | None = None,
) -> list[RequestResult]:
    """Send every job with at most config.concurrency in flight."""
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0

    async def _task(prompt: str, max_tokens: int) -> RequestResult:
        nonlocal completed
        async with semaphore:
            result = await send_request(session, config, prompt, max_tokens)
        completed += 1
        if on_progress:
            on_progress(completed, len(jobs))
        return result

    tasks = [asyncio.create_task(_task(prompt, max_tokens)) for prompt, max_tokens in jobs]
    return list(await asyncio.gather(*tasks))
