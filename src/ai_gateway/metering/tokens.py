"""Token estimation.

Exact counts need each vendor's tokenizer (and Anthropic/Gemini don't ship local ones), so the
gateway uses a cheap heuristic *before* a call (for TPM reservations) and prefers the
provider-reported `usage` afterwards. Estimated numbers are always flagged `estimated=True`.
"""

from __future__ import annotations

import math

from ai_gateway.models import ChatRequest

CHARS_PER_TOKEN = 4.0  # common rule of thumb for English text; deliberately conservative
PER_MESSAGE_OVERHEAD = 4  # role markers / separators added by chat templates


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_prompt_tokens(req: ChatRequest) -> int:
    total = 0
    for m in req.messages:
        total += PER_MESSAGE_OVERHEAD + estimate_text_tokens(m.content or "")
        for tc in m.tool_calls:
            total += estimate_text_tokens(tc.name) + estimate_text_tokens(tc.arguments)
    for tool in req.tools:
        total += estimate_text_tokens(tool.name) + estimate_text_tokens(str(tool.parameters))
    return total


def estimate_request_tokens(req: ChatRequest, default_max_tokens: int = 1024) -> int:
    """Upper-bound estimate used to *reserve* TPM before the call: prompt + max completion."""
    return estimate_prompt_tokens(req) + (req.max_tokens or default_max_tokens)
