"""Tool-call id generation for providers that don't return ids (Gemini, Ollama)."""

from __future__ import annotations

import uuid


def new_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"
