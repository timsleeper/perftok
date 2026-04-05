"""Shared fixtures for tokenflow tests."""

from __future__ import annotations

import json

import aiohttp
import pytest


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
