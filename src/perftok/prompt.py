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


def generate_prompt(target_tokens: int) -> str:
    """Generate a string that encodes to exactly *target_tokens* tokens (cl100k_base)."""
    if target_tokens <= 0:
        return ""

    # Build up words until we reach or exceed target, then trim.
    words: list[str] = []
    current_tokens = 0
    while current_tokens < target_tokens:
        word = random.choice(_WORD_POOL)
        words.append(word)
        current_tokens = len(_ENCODING.encode(" ".join(words)))

    # Trim from the end if we overshot.
    text = " ".join(words)
    tokens = _ENCODING.encode(text)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = _ENCODING.decode(tokens)
    elif len(tokens) < target_tokens:
        # Rare edge case — pad with single-token words.
        while len(tokens) < target_tokens:
            text += " a"
            tokens = _ENCODING.encode(text)
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            text = _ENCODING.decode(tokens)

    return text


def generate_output_token_count(mean: int, stddev: int) -> int:
    """Sample a positive output-token count from a normal distribution."""
    if stddev == 0:
        return max(1, mean)
    value = random.gauss(mean, stddev)
    return max(1, round(value))
