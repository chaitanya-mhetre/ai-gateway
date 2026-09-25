"""Cache facade used by the service: exact first, then (opt-in) semantic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ai_gateway.cache.exact import CacheStore, cache_key, exact_cache_allowed
from ai_gateway.cache.semantic import SemanticCache
from ai_gateway.config import CacheConfig
from ai_gateway.errors import GatewayError
from ai_gateway.models import ChatRequest, ChatResponse
from ai_gateway.observability.metrics import Metrics

log = logging.getLogger(__name__)


@dataclass
class CacheLookup:
    response: ChatResponse | None = None
    kind: str = "miss"  # miss | exact | semantic
    similarity: float | None = None
    exact_key: str | None = None
    semantic_vector: list[float] | None = None
    tenant: str = ""


class ResponseCache:
    def __init__(
        self,
        store: CacheStore,
        cfg: CacheConfig,
        metrics: Metrics,
        semantic: SemanticCache | None = None,
    ) -> None:
        self.store_ = store
        self.cfg = cfg
        self.metrics = metrics
        self.semantic = semantic

    async def lookup(self, tenant: str, req: ChatRequest, mode: str | None) -> CacheLookup:
        result = CacheLookup(tenant=tenant)
        if self.cfg.exact_enabled and exact_cache_allowed(req, mode):
            result.exact_key = cache_key(tenant, req)
            hit = await self.store_.get(result.exact_key)
            self.metrics.cache.labels("exact", "hit" if hit else "miss").inc()
            if hit is not None:
                result.response, result.kind = hit, "exact"
                return result
        if (
            self.semantic is not None
            and self.cfg.semantic_enabled
            and mode == "semantic"
            and not req.tools
        ):
            try:
                sem_hit, vector = await self.semantic.lookup(tenant, req)
            except GatewayError as exc:
                log.warning("semantic cache embedding failed, treating as miss: %s", exc.message)
                self.metrics.cache.labels("semantic", "error").inc()
                return result
            result.semantic_vector = vector
            self.metrics.cache.labels("semantic", "hit" if sem_hit else "miss").inc()
            if sem_hit is not None:
                result.response, result.kind, result.similarity = (
                    sem_hit.response,
                    "semantic",
                    sem_hit.similarity,
                )
        return result

    async def save(self, lookup: CacheLookup, req: ChatRequest, resp: ChatResponse) -> None:
        if resp.finish_reason not in ("stop", "length", "tool_calls"):
            return  # don't cache filtered/errored answers
        if lookup.exact_key is not None:
            await self.store_.set(lookup.exact_key, resp, self.cfg.exact_ttl_s)
        if lookup.semantic_vector is not None and self.semantic is not None:
            self.semantic.store(lookup.tenant, req, lookup.semantic_vector, resp)
