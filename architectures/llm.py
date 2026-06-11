"""Shared DeepSeek LLM client for all architectures.

DeepSeek exposes an OpenAI-compatible API. We use the openai SDK pointed
at DeepSeek's base URL. The client is lazily initialised and cached.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

# Load .env before anything reads os.getenv
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

from openai import OpenAI  # noqa: E402

MODEL    = "deepseek-chat"
BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# Pricing per million tokens (DeepSeek V3 cache-miss rates, June 2025)
_INPUT_PRICE_PER_M  = 0.27
_OUTPUT_PRICE_PER_M = 1.10

_client: OpenAI | None = None


def is_available() -> bool:
    return bool(os.getenv("DEEPSEEK_API_KEY"))


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. "
                "Add it to .env or set the environment variable."
            )
        _client = OpenAI(api_key=api_key, base_url=BASE_URL)
    return _client


def chat(
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[str, float, float]:
    """Single chat completion.

    Returns (answer_text, latency_ms, cost_usd).
    """
    client = get_client()
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    ms = (time.perf_counter() - t0) * 1000
    text  = resp.choices[0].message.content or ""
    usage = resp.usage
    cost  = (
        usage.prompt_tokens     * _INPUT_PRICE_PER_M  / 1_000_000
        + usage.completion_tokens * _OUTPUT_PRICE_PER_M / 1_000_000
    )
    return text.strip(), round(ms, 1), round(cost, 6)
