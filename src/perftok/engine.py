"""asyncio benchmark orchestrator with Semaphore-based concurrency."""

from __future__ import annotations

import asyncio
import time
import warnings

import aiohttp

from perftok.client import check_ssl, send_request
from perftok.models import BenchmarkConfig, BenchmarkReport, RequestResult
from perftok.prompt import generate_output_token_count, generate_prompt
from perftok.stats import compute_report


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkReport:
    """Run the full benchmark and return an aggregated report."""
    semaphore = asyncio.Semaphore(config.concurrency)
    results: list[RequestResult] = []

    async def _task(session: aiohttp.ClientSession) -> RequestResult:
        prompt = generate_prompt(config.mean_input_tokens)
        max_tokens = generate_output_token_count(
            config.mean_output_tokens, config.stddev_output_tokens
        )
        async with semaphore:
            return await send_request(session, config, prompt, max_tokens)

    if config.insecure:
        warnings.warn(
            "TLS/SSL certificate verification is disabled (--insecure).",
            UserWarning,
            stacklevel=2,
        )
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
        tasks = [asyncio.create_task(_task(session)) for _ in range(config.num_requests)]
        results = await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start
    return compute_report(list(results), total_duration_s=total_duration)
