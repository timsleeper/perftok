"""Tests for perftok.prompt — synthetic prompt generation."""

from __future__ import annotations

import pytest
import tiktoken

from perftok.prompt import generate_prompt, sample_token_count


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

    def test_does_not_retokenize_per_word(self, monkeypatch):
        """Generation must be linear: tokenize once, not once per appended word."""
        import perftok.prompt as prompt_mod

        calls = 0
        real_encode = prompt_mod._ENCODING.encode

        def counting_encode(text, *args, **kwargs):
            nonlocal calls
            calls += 1
            return real_encode(text, *args, **kwargs)

        monkeypatch.setattr(prompt_mod._ENCODING, "encode", counting_encode)

        generate_prompt(target_tokens=500)

        assert calls <= 3

    def test_returns_string(self):
        prompt = generate_prompt(target_tokens=10)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestSampleTokenCount:
    def test_mean_and_stddev(self):
        counts = [sample_token_count(mean=100, stddev=0) for _ in range(10)]
        assert all(c == 100 for c in counts)

    def test_always_positive(self):
        counts = [sample_token_count(mean=5, stddev=100) for _ in range(100)]
        assert all(c >= 1 for c in counts)

    def test_distribution_spread(self):
        counts = [sample_token_count(mean=500, stddev=100) for _ in range(200)]
        assert min(counts) < 500
        assert max(counts) > 500
