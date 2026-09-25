"""A fake OpenAI-compatible upstream with a fixed latency, used to measure gateway overhead."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import FastAPI

LATENCY_S = float(os.environ.get("MOCK_LATENCY_MS", "200")) / 1000
app = FastAPI()


@app.post("/v1/chat/completions")
async def chat(body: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(LATENCY_S)
    return {
        "id": "mock",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "m"),
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1},
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
