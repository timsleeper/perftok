"""asyncio benchmark orchestrator with Semaphore-based concurrency."""

from __future__ import annotations

import asyncio
import time

import aiohttp

from tokenflow.client import send_request
from tokenflow.models import BenchmarkConfig, BenchmarkReport, RequestResult
from tokenflow.prompt import generate_output_token_count, generate_prompt
from tokenflow.stats import compute_report


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

    start = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=config.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [asyncio.create_task(_task(session)) for _ in range(config.num_requests)]
        results = await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start
    return compute_report(list(results), total_duration_s=total_duration)
