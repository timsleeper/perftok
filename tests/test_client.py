"""Tests for tokenflow.client — SSE parsing, timing, error handling."""

from __future__ import annotations

import json

import aiohttp
import pytest
from aioresponses import aioresponses

from tests.conftest import make_ssl_error
from tokenflow.client import check_ssl, fetch_models, send_request
from tokenflow.models import BenchmarkConfig

BASE_URL = "http://test-server:8000"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"
MODELS_URL = f"{BASE_URL}/v1/models"


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


class TestFetchModels:
    @pytest.mark.asyncio
    async def test_returns_model_ids(self):
        payload = {
            "object": "list",
            "data": [
                {"id": "model-a", "object": "model"},
                {"id": "model-b", "object": "model"},
            ],
        }
        with aioresponses() as m:
            m.get(MODELS_URL, payload=payload)
            models = await fetch_models(BASE_URL)
        assert models == ["model-a", "model-b"]

    @pytest.mark.asyncio
    async def test_single_model(self):
        payload = {
            "object": "list",
            "data": [{"id": "only-model", "object": "model"}],
        }
        with aioresponses() as m:
            m.get(MODELS_URL, payload=payload)
            models = await fetch_models(BASE_URL)
        assert models == ["only-model"]

    @pytest.mark.asyncio
    async def test_with_api_key(self):
        payload = {"object": "list", "data": [{"id": "m", "object": "model"}]}
        with aioresponses() as m:
            m.get(MODELS_URL, payload=payload)
            models = await fetch_models(BASE_URL, api_key="sk-test")
        assert models == ["m"]

    @pytest.mark.asyncio
    async def test_http_error_raises(self):
        with aioresponses() as m:
            m.get(MODELS_URL, status=401)
            with pytest.raises(RuntimeError, match="401"):
                await fetch_models(BASE_URL)

    @pytest.mark.asyncio
    async def test_connection_error_raises(self):
        with aioresponses() as m:
            m.get(MODELS_URL, exception=aiohttp.ClientConnectionError("refused"))
            with pytest.raises(RuntimeError, match="connect"):
                await fetch_models(BASE_URL)

    @pytest.mark.asyncio
    async def test_empty_models_raises(self):
        payload = {"object": "list", "data": []}
        with aioresponses() as m:
            m.get(MODELS_URL, payload=payload)
            with pytest.raises(RuntimeError, match="[Nn]o models"):
                await fetch_models(BASE_URL)

    @pytest.mark.asyncio
    async def test_ssl_error_warns_and_retries(self):
        """fetch_models retries with ssl=False on SSL failure."""
        https_url = "https://local-gpu:8000"
        models_url = f"{https_url}/v1/models"
        payload = {
            "object": "list",
            "data": [{"id": "local-llama", "object": "model"}],
        }
        with aioresponses() as m:
            m.get(models_url, exception=make_ssl_error())  # first attempt
            m.get(models_url, payload=payload)  # retry with ssl=False
            with pytest.warns(UserWarning, match="TLS/SSL"):
                models = await fetch_models(https_url)
        assert models == ["local-llama"]


HTTPS_URL = "https://local-gpu:8000"
HTTPS_MODELS_URL = f"{HTTPS_URL}/v1/models"


class TestCheckSsl:
    @pytest.mark.asyncio
    async def test_http_url_skips_check(self):
        result = await check_ssl("http://localhost:8000")
        assert result is True

    @pytest.mark.asyncio
    async def test_https_valid_cert(self):
        with aioresponses() as m:
            m.get(HTTPS_MODELS_URL, status=200)
            result = await check_ssl(HTTPS_URL)
        assert result is True

    @pytest.mark.asyncio
    async def test_https_ssl_failure_returns_false_and_warns(self):
        with aioresponses() as m:
            m.get(HTTPS_MODELS_URL, exception=make_ssl_error())
            with pytest.warns(UserWarning, match="TLS/SSL"):
                result = await check_ssl(HTTPS_URL)
        assert result is False

    @pytest.mark.asyncio
    async def test_https_auth_error_still_returns_true(self):
        """SSL check passes even if server returns 401 — connection worked."""
        with aioresponses() as m:
            m.get(HTTPS_MODELS_URL, status=401)
            result = await check_ssl(HTTPS_URL)
        assert result is True
