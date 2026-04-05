"""aiohttp streaming SSE client with TTFT/ITL measurement."""

from __future__ import annotations

import json
import time

import aiohttp

from tokenflow.models import BenchmarkConfig, RequestResult


async def send_request(
    session: aiohttp.ClientSession,
    config: BenchmarkConfig,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a single chat completion request and measure latencies."""
    url = f"{config.url}/v1/chat/completions"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    payload: dict = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": config.streaming,
    }

    start = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                elapsed = (time.perf_counter() - start) * 1000
                return RequestResult(
                    success=False,
                    e2e_latency_ms=elapsed,
                    error=f"HTTP {resp.status}: {body[:200]}",
                )

            if config.streaming:
                return await _handle_streaming(resp, start)
            else:
                return await _handle_non_streaming(resp, start)

    except TimeoutError:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(
            success=False, e2e_latency_ms=elapsed, error="Request timeout"
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - start) * 1000
        return RequestResult(
            success=False, e2e_latency_ms=elapsed, error=str(exc)
        )


async def _handle_streaming(
    resp: aiohttp.ClientResponse,
    start: float,
) -> RequestResult:
    """Parse SSE stream, measure TTFT and inter-token latencies."""
    ttft: float | None = None
    token_times: list[float] = []
    output_tokens = 0

    async for line_bytes in resp.content:
        line = line_bytes.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            now = time.perf_counter()
            if ttft is None:
                ttft = (now - start) * 1000
            token_times.append(now)
            output_tokens += 1

    end = time.perf_counter()
    e2e = (end - start) * 1000

    itl: list[float] = []
    for i in range(1, len(token_times)):
        itl.append((token_times[i] - token_times[i - 1]) * 1000)

    return RequestResult(
        success=True,
        ttft_ms=ttft,
        e2e_latency_ms=e2e,
        output_tokens=output_tokens,
        inter_token_latencies_ms=itl,
    )


async def _handle_non_streaming(
    resp: aiohttp.ClientResponse,
    start: float,
) -> RequestResult:
    """Parse non-streaming response, extract usage."""
    body = await resp.json()
    end = time.perf_counter()
    e2e = (end - start) * 1000

    usage = body.get("usage", {})
    output_tokens = usage.get("completion_tokens", 0)

    return RequestResult(
        success=True,
        ttft_ms=e2e,  # For non-streaming, TTFT ≈ full response time
        e2e_latency_ms=e2e,
        output_tokens=output_tokens,
    )
