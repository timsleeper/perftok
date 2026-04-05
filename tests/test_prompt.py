"""Tests for tokenflow.prompt — synthetic prompt generation."""

from __future__ import annotations

import tiktoken
import pytest

from tokenflow.prompt import generate_prompt, generate_output_token_count


@pytest.fixture
def encoding():
    return tiktoken.get_encoding("cl100k_base")


class TestGeneratePrompt:
    def test_exact_token_count(self, encoding):
        prompt = generate_prompt(target_tokens=100)
        actual = len(encoding.encode(prompt))
        assert actual == 100

    def test_small_token_count(self, encoding):
        prompt = generate_prompt(target_tokens=1)
        actual = len(encoding.encode(prompt))
        assert actual == 1

    def test_large_token_count(self, encoding):
        prompt = generate_prompt(target_tokens=1000)
        actual = len(encoding.encode(prompt))
        assert actual == 1000

    def test_returns_string(self):
        prompt = generate_prompt(target_tokens=10)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestGenerateOutputTokenCount:
    def test_mean_and_stddev(self):
        counts = [generate_output_token_count(mean=100, stddev=0) for _ in range(10)]
        assert all(c == 100 for c in counts)

    def test_always_positive(self):
        counts = [generate_output_token_count(mean=5, stddev=100) for _ in range(100)]
        assert all(c >= 1 for c in counts)

    def test_distribution_spread(self):
        counts = [generate_output_token_count(mean=500, stddev=100) for _ in range(200)]
        assert min(counts) < 500
        assert max(counts) > 500
