"""asyncio benchmark orchestrator with Semaphore-based concurrency."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import aiohttp
import click

from perftok.client import check_ssl, send_request
from perftok.models import BenchmarkConfig, BenchmarkReport, RequestResult
from perftok.prompt import generate_output_token_count, generate_prompt
from perftok.stats import compute_report

if TYPE_CHECKING:
    pass


async def run_benchmark(
    config: BenchmarkConfig,
    on_progress: Callable[[int, int], None] | None = None,
) -> BenchmarkReport:
    """Run the full benchmark and return an aggregated report."""
    semaphore = asyncio.Semaphore(config.concurrency)
    completed = 0
    lock = asyncio.Lock()

    async def _task(session: aiohttp.ClientSession) -> RequestResult:
        nonlocal completed
        prompt = generate_prompt(config.mean_input_tokens)
        max_tokens = generate_output_token_count(
            config.mean_output_tokens, config.stddev_output_tokens
        )
        async with semaphore:
            result = await send_request(session, config, prompt, max_tokens)
        async with lock:
            completed += 1
            if on_progress:
                on_progress(completed, config.num_requests)
        return result

    if config.insecure:
        click.echo("TLS/SSL certificate verification is disabled (--insecure).")
        ssl_param: bool | None = False  # noqa: S507
    else:
        await check_ssl(config.url, config.api_key)
        ssl_param = None

    connector = aiohttp.TCPConnector(ssl=ssl_param)

    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=config.timeout)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout
    ) as session:
        tasks = [
            asyncio.create_task(_task(session))
            for _ in range(config.num_requests)
        ]
        results = await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start
    return compute_report(list(results), total_duration_s=total_duration)
