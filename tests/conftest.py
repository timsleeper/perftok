"""Shared fixtures for perftok tests."""

from __future__ import annotations

import asyncio
import json

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer


def make_ssl_error() -> aiohttp.ClientConnectorSSLError:
    """Create a testable SSL error, bypassing complex aiohttp internals."""
    err = aiohttp.ClientConnectorSSLError.__new__(aiohttp.ClientConnectorSSLError)
    err.args = ("SSL: CERTIFICATE_VERIFY_FAILED",)
    return err


def make_sse_chunk(
    content: str = "",
    finish_reason: str | None = None,
    model: str = "test-model",
) -> str:
    """Build a single SSE data line for a streaming chat completion chunk."""
    delta: dict = {}
    if content:
        delta["content"] = content
    choice: dict = {"index": 0, "delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [choice],
    }
    return f"data: {json.dumps(payload)}\n\n"


def make_sse_done() -> str:
    """Build the SSE stream termination line."""
    return "data: [DONE]\n\n"


def make_completion_response(
    content: str = "Hello, world!",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    model: str = "test-model",
) -> dict:
    """Build a non-streaming chat completion response body."""
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@pytest.fixture
def sse_chunk_factory():
    """Factory fixture for SSE chunks."""
    return make_sse_chunk


@pytest.fixture
def completion_response_factory():
    """Factory fixture for completion responses."""
    return make_completion_response


# --- Real local HTTP server fixtures -------------------------------------------


def sse_handler(body: str, *, chunk_delay_s: float = 0.0, status: int = 200):
    """Return a handler that streams *body* line-by-line as text/event-stream."""

    async def handler(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=status, headers={"Content-Type": "text/event-stream"}
        )
        await resp.prepare(request)
        for line in body.splitlines(keepends=True):
            if chunk_delay_s:
                await asyncio.sleep(chunk_delay_s)
            await resp.write(line.encode())
        return resp

    return handler


@pytest.fixture
async def make_server():
    """Start real aiohttp servers for tests. Returns a factory: (routes) -> base_url."""
    servers: list[TestServer] = []

    async def _make(routes: dict[str, object]) -> str:
        app = web.Application()
        for key, handler in routes.items():
            method, path = key.split(" ", 1)
            app.router.add_route(method, path, handler)
        server = TestServer(app)
        await server.start_server()
        servers.append(server)
        return str(server.make_url("")).rstrip("/")

    yield _make
    for server in servers:
        await server.close()


@pytest.fixture
def https_alias(monkeypatch):
    """Route requests for a fake https origin to a real http test server.

    Returns a function ``alias(https_origin, http_base_url)``. Every request the
    client makes to *https_origin* is transparently sent to *http_base_url*
    instead, and the ``ssl`` kwarg each request was made with is recorded in
    the returned list ``alias.ssl_args``.
    """
    original = aiohttp.ClientSession._request
    mapping: dict[str, str] = {}
    ssl_args: list[object] = []

    async def patched(self, method, url, **kwargs):
        url = str(url)
        for origin, target in mapping.items():
            if url.startswith(origin):
                ssl_args.append(kwargs.get("ssl"))
                url = target + url[len(origin):]
        return await original(self, method, url, **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "_request", patched)

    def alias(https_origin: str, http_base_url: str) -> None:
        mapping[https_origin] = http_base_url

    alias.ssl_args = ssl_args  # type: ignore[attr-defined]
    return alias
