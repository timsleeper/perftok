"""Synthetic prompt generation with tiktoken."""

from __future__ import annotations

import random

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

# A pool of common English words that each encode to a single token in cl100k_base.
_WORD_POOL = [
    "the", "of", "and", "to", "in", "is", "it", "for", "that", "was",
    "on", "are", "as", "with", "they", "be", "at", "one", "have", "this",
    "from", "or", "had", "by", "not", "but", "what", "all", "were", "we",
    "when", "your", "can", "said", "there", "use", "an", "each", "which",
    "she", "do", "how", "their", "if", "will", "up", "about", "out", "many",
]


def count_tokens(text: str) -> int:
    """Return the number of cl100k_base tokens in *text*."""
    return len(_ENCODING.encode(text))


def generate_prompt(target_tokens: int) -> str:
    """Generate a string that encodes to exactly *target_tokens* tokens (cl100k_base)."""
    if target_tokens <= 0:
        return ""

    # Every pool word is a single token, so target_tokens words lands on or very
    # near the target. Tokenize once and then correct, rather than re-encoding
    # the growing string after every word (quadratic in prompt length).
    words = random.choices(_WORD_POOL, k=target_tokens)
    text = " ".join(words)
    tokens = _ENCODING.encode(text)

    if len(tokens) < target_tokens:
        # Rare edge case — pad with single-token words.
        text += " a" * (target_tokens - len(tokens))
        tokens = _ENCODING.encode(text)

    if len(tokens) > target_tokens:
        text = _ENCODING.decode(tokens[:target_tokens])

    return text


def generate_output_token_count(mean: int, stddev: int) -> int:
    """Sample a positive output-token count from a normal distribution."""
    if stddev == 0:
        return max(1, mean)
    value = random.gauss(mean, stddev)
    return max(1, round(value))
