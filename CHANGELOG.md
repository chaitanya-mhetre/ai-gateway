# Changelog

## [0.1.0] - 2026-09-25
Built milestone by milestone (git tags `m1`…`m7`).

### M1 - Unified API
- OpenAI-compatible `/v1/chat/completions` (JSON + SSE), `/v1/embeddings`, `/v1/models`
- Provider-neutral internal model; scriptable `MockProvider`; OpenAI adapter on raw httpx
- Compatibility test with the official `openai` SDK

### M2 - All adapters
- Anthropic Messages, Gemini generateContent (+SSE), Ollama (NDJSON), tool-call translation
- Fixture-based contract tests per provider

### M3 - Reliability
- Router policies (priority, weighted, cost, latency) with capability/context filtering
- Executor: overall deadline, retries with full jitter and Retry-After, configurable fallback
- Circuit breaker (in-memory and Redis Lua); stream failover only before the first token

### M4 - Keys and limits
- Hashed API keys (HMAC + pepper), admin API, per-key alias allow-lists
- Lua token-bucket RPM/TPM with reserve/reconcile; monthly token/cost budgets

### M5 - Metering and observability
- Versioned price table, usage events via Redis Streams + batch worker + daily rollups, usage API
- Prometheus metrics, OpenTelemetry spans, Grafana dashboard

### M6 - Caching
- Tenant-scoped exact cache (memory LRU / Redis); opt-in semantic cache with similarity threshold

### M7 - Release
- Multi-stage Dockerfile, compose stack, CI, measured overhead benchmark, docs and learning guide
