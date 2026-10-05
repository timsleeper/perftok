"""Tests for perftok.client — SSE parsing, timing, error handling.

All HTTP tests run against a real local aiohttp server (see conftest.make_server).
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import aiohttp
import pytest
import tiktoken
from aiohttp import web

from perftok.client import check_ssl, fetch_models, send_request
from perftok.models import BenchmarkConfig
from tests.conftest import make_completion_response, make_ssl_error, sse_handler

CHAT = "POST /v1/chat/completions"
MODELS = "GET /v1/models"
HTTPS_URL = "https://local-gpu:8000"


def _config(url: str, **overrides) -> BenchmarkConfig:
    defaults = dict(model="test-model", url=url, streaming=True, timeout=10)
    defaults.update(overrides)
    return BenchmarkConfig(**defaults)


def _chunk(content: str = "", finish_reason: str | None = None) -> dict:
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


def _sse(*payloads: dict, done: bool = True) -> str:
    body = "".join(f"data: {json.dumps(p)}\n\n" for p in payloads)
    if done:
        body += "data: [DONE]\n\n"
    return body


def _json_handler(payload: dict | None = None, status: int = 200, text: str | None = None):
    async def handler(request: web.Request) -> web.Response:
        if text is not None:
            return web.Response(status=status, text=text)
        return web.json_response(payload if payload is not None else {}, status=status)

    return handler


async def _send(url: str, config: BenchmarkConfig | None = None, **kw):
    config = config or _config(url)
    async with aiohttp.ClientSession() as session:
        return await send_request(
            session=session, config=config, prompt="test prompt", max_tokens=100, **kw
        )


class TestStreamingRequest:
    async def test_successful_streaming(self, make_server):
        body = _sse(_chunk("Hello"), _chunk(" world"), _chunk("!", "stop"))
        url = await make_server({CHAT: sse_handler(body)})

        result = await _send(url)

        assert result.success is True
        assert result.output_tokens == 3
        assert result.ttft_ms is not None
        assert result.ttft_ms > 0
        assert result.e2e_latency_ms > 0
        assert len(result.inter_chunk_latencies_ms) == 2

    async def test_itl_is_decode_time_per_token(self, make_server):
        """ITL follows aiperf: (latency - TTFT) / (output_tokens - 1)."""
        body = _sse(_chunk("Hello"), _chunk(" world"), _chunk("!", "stop"))
        url = await make_server({CHAT: sse_handler(body, chunk_delay_s=0.02)})

        result = await _send(url)

        expected = (result.e2e_latency_ms - result.ttft_ms) / (result.output_tokens - 1)
        assert result.inter_token_latency_ms == pytest.approx(expected)

    async def test_itl_uses_usage_token_count_not_chunk_count(self, make_server):
        """A single multi-token chunk still yields a per-token ITL."""
        usage_chunk = {"choices": [], "usage": {"completion_tokens": 11}}
        body = _sse(_chunk("Hello"), _chunk(" world, how are you doing", "stop"), usage_chunk)
        url = await make_server({CHAT: sse_handler(body, chunk_delay_s=0.02)})

        result = await _send(url)

        expected = (result.e2e_latency_ms - result.ttft_ms) / 10
        assert result.inter_token_latency_ms == pytest.approx(expected)
        assert len(result.inter_chunk_latencies_ms) == 1

    async def test_streaming_single_token(self, make_server):
        url = await make_server({CHAT: sse_handler(_sse(_chunk("Hi", "stop")))})

        result = await _send(url)

        assert result.success is True
        assert result.output_tokens == 1
        assert result.inter_chunk_latencies_ms == []
        assert result.inter_token_latency_ms is None

    async def test_ttft_reflects_first_chunk_arrival(self, make_server):
        """TTFT measures time to the first content chunk, not the whole stream."""
        body = _sse(_chunk("a"), _chunk("b"), _chunk("c", "stop"))
        url = await make_server({CHAT: sse_handler(body, chunk_delay_s=0.05)})

        result = await _send(url)

        # 4 delayed lines before first content line (data, blank) = ~2 * 50ms
        assert result.ttft_ms < result.e2e_latency_ms / 2

    async def test_output_tokens_from_usage_chunk(self, make_server):
        """When the server reports usage, completion_tokens wins over chunk counting."""
        usage_chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 42, "total_tokens": 52},
        }
        body = _sse(_chunk("Hello world, how are you"), _chunk("", "stop"), usage_chunk)
        url = await make_server({CHAT: sse_handler(body)})

        result = await _send(url)

        assert result.success is True
        assert result.output_tokens == 42

    async def test_output_tokens_tokenized_when_no_usage(self, make_server):
        """Without usage, output tokens are counted with the tokenizer, not per chunk."""
        text_a, text_b = "Hello", " wonderful world of benchmarking"
        body = _sse(_chunk(text_a), _chunk(text_b, "stop"))
        url = await make_server({CHAT: sse_handler(body)})

        result = await _send(url)

        expected = len(tiktoken.get_encoding("cl100k_base").encode(text_a + text_b))
        assert expected > 2  # proves this is not a chunk count
        assert result.output_tokens == expected

    async def test_streaming_request_asks_for_usage(self, make_server):
        """Streaming requests set stream_options.include_usage so servers report usage."""
        seen: list[dict] = []

        async def handler(request: web.Request) -> web.StreamResponse:
            seen.append(await request.json())
            return await sse_handler(_sse(_chunk("x", "stop")))(request)

        url = await make_server({CHAT: handler})
        await _send(url)

        assert seen[0]["stream"] is True
        assert seen[0]["stream_options"] == {"include_usage": True}

    async def test_ignore_eos_sends_ignore_eos_and_min_tokens(self, make_server):
        seen: list[dict] = []

        async def handler(request: web.Request) -> web.StreamResponse:
            seen.append(await request.json())
            return await sse_handler(_sse(_chunk("x", "stop")))(request)

        url = await make_server({CHAT: handler})
        await _send(url, _config(url, ignore_eos=True))

        assert seen[0]["ignore_eos"] is True
        assert seen[0]["min_tokens"] == seen[0]["max_tokens"]

    async def test_ignore_eos_off_sends_neither(self, make_server):
        seen: list[dict] = []

        async def handler(request: web.Request) -> web.StreamResponse:
            seen.append(await request.json())
            return await sse_handler(_sse(_chunk("x", "stop")))(request)

        url = await make_server({CHAT: handler})
        await _send(url)

        assert "ignore_eos" not in seen[0]
        assert "min_tokens" not in seen[0]

    async def test_result_records_requested_output_tokens(self, make_server):
        url = await make_server({CHAT: sse_handler(_sse(_chunk("x", "stop")))})

        result = await _send(url)

        assert result.requested_output_tokens == 100  # max_tokens passed by _send

    async def test_non_streaming_request_omits_stream_options(self, make_server):
        seen: list[dict] = []

        async def handler(request: web.Request) -> web.Response:
            seen.append(await request.json())
            return web.json_response(make_completion_response())

        url = await make_server({CHAT: handler})
        await _send(url, _config(url, streaming=False))

        assert seen[0]["stream"] is False
        assert "stream_options" not in seen[0]


class TestNonStreamingRequest:
    async def test_successful_non_streaming(self, make_server):
        payload = make_completion_response(content="Hello world!", completion_tokens=3)
        url = await make_server({CHAT: _json_handler(payload)})

        result = await _send(url, _config(url, streaming=False))

        assert result.success is True
        assert result.output_tokens == 3
        assert result.e2e_latency_ms > 0
        assert result.ttft_ms is not None
        assert result.inter_token_latency_ms is None  # undefined without streaming


class TestErrorHandling:
    async def test_http_error_no_body_leakage(self, make_server):
        """Error message includes status code but NOT the response body."""
        url = await make_server(
            {CHAT: _json_handler(status=500, text="secret internal stack trace")}
        )

        result = await _send(url)

        assert result.success is False
        assert "500" in result.error
        assert "secret" not in result.error

    async def test_connection_error(self, unused_tcp_port):
        result = await _send(f"http://127.0.0.1:{unused_tcp_port}")

        assert result.success is False
        assert result.error is not None

    async def test_timeout_error(self, make_server):
        async def slow(request: web.Request) -> web.Response:
            await asyncio.sleep(2)
            return web.json_response({})

        url = await make_server({CHAT: slow})

        result = await _send(url, _config(url, timeout=1))

        assert result.success is False
        assert "timeout" in result.error.lower()


class TestFetchModels:
    @staticmethod
    def _models(*ids: str) -> dict:
        return {"object": "list", "data": [{"id": i, "object": "model"} for i in ids]}

    async def test_returns_model_ids(self, make_server):
        url = await make_server({MODELS: _json_handler(self._models("model-a", "model-b"))})
        assert await fetch_models(url) == ["model-a", "model-b"]

    async def test_single_model(self, make_server):
        url = await make_server({MODELS: _json_handler(self._models("only-model"))})
        assert await fetch_models(url) == ["only-model"]

    async def test_with_api_key_sends_bearer(self, make_server):
        seen: list[str | None] = []

        async def handler(request: web.Request) -> web.Response:
            seen.append(request.headers.get("Authorization"))
            return web.json_response(self._models("m"))

        url = await make_server({MODELS: handler})

        assert await fetch_models(url, api_key="sk-test") == ["m"]
        assert seen == ["Bearer sk-test"]

    async def test_http_error_no_body_leakage(self, make_server):
        """Error includes status code but NOT response body."""
        url = await make_server({MODELS: _json_handler(status=401, text="secret token info")})

        with pytest.raises(RuntimeError, match="401") as exc_info:
            await fetch_models(url)
        assert "secret" not in str(exc_info.value)

    async def test_connection_error_raises(self, unused_tcp_port):
        with pytest.raises(RuntimeError, match="connect"):
            await fetch_models(f"http://127.0.0.1:{unused_tcp_port}")

    async def test_empty_models_raises(self, make_server):
        url = await make_server({MODELS: _json_handler(self._models())})
        with pytest.raises(RuntimeError, match="[Nn]o models"):
            await fetch_models(url)

    async def test_ssl_error_without_insecure_raises(self):
        """SSL failure without insecure=True raises with --insecure hint."""
        with patch.object(aiohttp.ClientSession, "_request", side_effect=make_ssl_error()):
            with pytest.raises(RuntimeError, match="--insecure"):
                await fetch_models(HTTPS_URL)

    async def test_ssl_with_insecure_disables_verification(self, make_server, https_alias):
        url = await make_server({MODELS: _json_handler(self._models("local-llama"))})
        https_alias(HTTPS_URL, url)

        assert await fetch_models(HTTPS_URL, insecure=True) == ["local-llama"]
        assert https_alias.ssl_args == [False]


class TestCheckSsl:
    async def test_http_url_skips_check(self):
        await check_ssl("http://localhost:8000")  # should not raise

    async def test_https_valid_cert(self, make_server, https_alias):
        url = await make_server({MODELS: _json_handler({})})
        https_alias(HTTPS_URL, url)

        await check_ssl(HTTPS_URL)  # should not raise

        assert https_alias.ssl_args == [None]  # default verification left on

    async def test_https_ssl_failure_raises_with_hint(self):
        with patch.object(aiohttp.ClientSession, "_request", side_effect=make_ssl_error()):
            with pytest.raises(RuntimeError, match="--insecure"):
                await check_ssl(HTTPS_URL)

    async def test_https_auth_error_passes(self, make_server, https_alias):
        """SSL check passes even if server returns 401 — connection worked."""
        url = await make_server({MODELS: _json_handler(status=401, text="nope")})
        https_alias(HTTPS_URL, url)

        await check_ssl(HTTPS_URL)  # should not raise
