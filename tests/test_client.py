"""Tests for tokenflow.client — SSE parsing, timing, error handling."""

from __future__ import annotations

import json

import aiohttp
import pytest
from aioresponses import aioresponses

from tokenflow.client import send_request
from tokenflow.models import BenchmarkConfig

BASE_URL = "http://test-server:8000"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"


@pytest.fixture
def config():
    return BenchmarkConfig(model="test-model", url=BASE_URL, streaming=True, timeout=10)


@pytest.fixture
def config_non_streaming():
    return BenchmarkConfig(
        model="test-model", url=BASE_URL, streaming=False, timeout=10
    )


def _sse_body(chunks: list[str]) -> str:
    body = ""
    for c in chunks:
        body += f"data: {json.dumps(c) if isinstance(c, dict) else c}\n\n"
    return body


def _make_chunk(content: str = "", finish_reason: str | None = None) -> dict:
    delta = {}
    if content:
        delta["content"] = content
    choice = {"index": 0, "delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "test-model",
        "choices": [choice],
    }


class TestStreamingRequest:
    @pytest.mark.asyncio
    async def test_successful_streaming(self, config):
        chunks = [
            _make_chunk("Hello"),
            _make_chunk(" world"),
            _make_chunk("!", "stop"),
        ]
        body = ""
        for c in chunks:
            body += f"data: {json.dumps(c)}\n\n"
        body += "data: [DONE]\n\n"

        with aioresponses() as m:
            m.post(CHAT_URL, status=200, body=body, content_type="text/event-stream")
            async with aiohttp.ClientSession() as session:
                result = await send_request(
                    session=session,
                    config=config,
                    prompt="test prompt",
                    max_tokens=100,
                )
        assert result.success is True
        assert result.output_tokens == 3
        assert result.ttft_ms is not None
        assert result.ttft_ms > 0
        assert result.e2e_latency_ms > 0
        assert len(result.inter_token_latencies_ms) == 2  # between 3 tokens

    @pytest.mark.asyncio
    async def test_streaming_single_token(self, config):
        body = f"data: {json.dumps(_make_chunk('Hi', 'stop'))}\n\ndata: [DONE]\n\n"

        with aioresponses() as m:
            m.post(CHAT_URL, status=200, body=body, content_type="text/event-stream")
            async with aiohttp.ClientSession() as session:
                result = await send_request(
                    session=session, config=config, prompt="test", max_tokens=10
                )
        assert result.success is True
        assert result.output_tokens == 1
        assert result.inter_token_latencies_ms == []


class TestNonStreamingRequest:
    @pytest.mark.asyncio
    async def test_successful_non_streaming(self, config_non_streaming):
        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello world!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 3,
                "total_tokens": 13,
            },
        }

        with aioresponses() as m:
            m.post(CHAT_URL, status=200, payload=response_body)
            async with aiohttp.ClientSession() as session:
                result = await send_request(
                    session=session,
                    config=config_non_streaming,
                    prompt="test",
                    max_tokens=100,
                )
        assert result.success is True
        assert result.output_tokens == 3
        assert result.e2e_latency_ms > 0
        assert result.ttft_ms is not None


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_http_error(self, config):
        with aioresponses() as m:
            m.post(CHAT_URL, status=500)
            async with aiohttp.ClientSession() as session:
                result = await send_request(
                    session=session, config=config, prompt="test", max_tokens=10
                )
        assert result.success is False
        assert result.error is not None
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_connection_error(self, config):
        with aioresponses() as m:
            m.post(CHAT_URL, exception=aiohttp.ClientConnectionError("refused"))
            async with aiohttp.ClientSession() as session:
                result = await send_request(
                    session=session, config=config, prompt="test", max_tokens=10
                )
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_timeout_error(self, config):
        with aioresponses() as m:
            m.post(CHAT_URL, exception=TimeoutError())
            async with aiohttp.ClientSession() as session:
                result = await send_request(
                    session=session, config=config, prompt="test", max_tokens=10
                )
        assert result.success is False
        assert "timeout" in result.error.lower()
