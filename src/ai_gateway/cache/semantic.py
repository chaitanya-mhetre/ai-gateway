"""Semantic cache (opt-in): reuse an answer if a *similar enough* prompt was answered before.

Similarity = cosine similarity of prompt embeddings, obtained through the gateway's own
embedding alias. The danger is **false hits**: "What's the refund window for India?" and
"...for the US?" can embed very close together, yet need different answers. Hence:
- opt-in per request (`X-Gateway-Cache: semantic`) and per deployment (`cache.semantic_enabled`);
- a high threshold (default 0.95), and every hit reports its similarity score;
- per-tenant, per-alias indexes (never cross-tenant);
- only for requests without tools.

Storage is an in-memory brute-force index (O(n) per lookup, bounded size). That's fine for a
demo and small deployments. Production would use pgvector or a Redis vector index (see README).
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ai_gateway.models import ChatRequest, ChatResponse

Embedder = Callable[[str], Awaitable[list[float]]]


def prompt_text(req: ChatRequest) -> str:
    return "\n".join(f"{m.role}: {m.content or ''}" for m in req.messages)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


@dataclass
class SemanticHit:
    response: ChatResponse
    similarity: float


class SemanticCache:
    def __init__(
        self, embed: Embedder, *, threshold: float = 0.95, max_entries_per_index: int = 1000
    ) -> None:
        self.embed = embed
        self.threshold = threshold
        self.max_entries = max_entries_per_index
        self._indexes: dict[tuple[str, str], deque[tuple[list[float], ChatResponse]]] = {}

    async def lookup(self, tenant: str, req: ChatRequest) -> tuple[SemanticHit | None, list[float]]:
        """Returns (best hit above threshold or None, the query vector for a later `store`)."""
        vector = await self.embed(prompt_text(req))
        best: SemanticHit | None = None
        for vec, resp in self._indexes.get((tenant, req.model), ()):
            sim = cosine(vector, vec)
            if sim >= self.threshold and (best is None or sim > best.similarity):
                best = SemanticHit(resp, sim)
        return best, vector

    def store(
        self, tenant: str, req: ChatRequest, vector: list[float], response: ChatResponse
    ) -> None:
        index = self._indexes.setdefault((tenant, req.model), deque(maxlen=self.max_entries))
        index.append((vector, response))
