"""Live smoke tests against real providers. MANUAL ONLY: `uv run pytest -m live`.

Each test makes one tiny call (max_tokens=16) and is skipped when its key or model env var is
missing. Model names come from env vars because they change over time.
"""

from __future__ import annotations

import os

import pytest

from ai_gateway.models import ChatRequest, Message
from ai_gateway.providers.anthropic import AnthropicProvider
from ai_gateway.providers.base import Provider
from ai_gateway.providers.gemini import GeminiProvider
from ai_gateway.providers.openai import OpenAIProvider

CASES = [
    (
        "OPENAI_API_KEY",
        "LIVE_OPENAI_MODEL",
        lambda k: OpenAIProvider("openai", "https://api.openai.com/v1", k),
    ),
    (
        "ANTHROPIC_API_KEY",
        "LIVE_ANTHROPIC_MODEL",
        lambda k: AnthropicProvider("anthropic", "https://api.anthropic.com", k),
    ),
    (
        "GEMINI_API_KEY",
        "LIVE_GEMINI_MODEL",
        lambda k: GeminiProvider("gemini", "https://generativelanguage.googleapis.com/v1beta", k),
    ),
]


@pytest.mark.live
@pytest.mark.parametrize(("key_env", "model_env", "factory"), CASES)
async def test_live_chat_and_stream(key_env: str, model_env: str, factory: object) -> None:
    key, model = os.environ.get(key_env), os.environ.get(model_env)
    if not key or not model:
        pytest.skip(f"set {key_env} and {model_env}")
    provider: Provider = factory(key)  # type: ignore[operator]
    req = ChatRequest(
        model=model, messages=[Message(role="user", content="Say OK.")], max_tokens=16
    )
    resp = await provider.chat(req, model, 30)
    assert resp.content and resp.usage.completion_tokens > 0
    chunks = [c async for c in provider.stream(req.model_copy(update={"stream": True}), model, 30)]
    assert any(c.content for c in chunks)
    await provider.aclose()
