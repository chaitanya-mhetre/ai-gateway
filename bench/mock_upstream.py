"""A fake OpenAI-compatible upstream with a fixed latency, used by the benchmarks.

Fault injection for `bench/outage.py`: `POST /_fault {"mode": "ok" | "hang" | "5xx"}` switches the
behaviour at runtime. `hang` accepts the request and never answers in time (a stuck provider);
`5xx` answers 503 immediately. A hard crash is simulated by killing the process instead.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

LATENCY_S = float(os.environ.get("MOCK_LATENCY_MS", "200")) / 1000
HANG_S = 120.0
app = FastAPI()
_state = {"mode": "ok"}


@app.post("/_fault")
async def set_fault(body: dict[str, str]) -> dict[str, str]:
    _state["mode"] = body["mode"]
    return _state


@app.post("/v1/chat/completions", response_model=None)
async def chat(body: dict[str, Any]) -> dict[str, Any] | JSONResponse:
    mode = _state["mode"]
    if mode == "5xx":
        return JSONResponse({"error": {"message": "injected outage"}}, status_code=503)
    await asyncio.sleep(HANG_S if mode == "hang" else LATENCY_S)
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
