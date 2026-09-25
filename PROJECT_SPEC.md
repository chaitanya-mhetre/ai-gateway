# AI Gateway
> A self-hosted LLM gateway. It puts OpenAI, Anthropic, Gemini and local models (Ollama) behind one API, with configurable routing, fallback, caching, quotas, and token/cost/latency observability.

## 1. Problem & why it exists
Once a company has more than one LLM feature, the same problems appear in every service:
- **Vendor lock-in and outages.** When one provider has a bad hour, every feature built on it breaks.
- **No cost visibility.** Finance asks "what did the AI features cost last month, per team?" and nobody knows.
- **Duplicated plumbing.** Every team re-implements retries, timeouts, key management, rate limits and logging.
- **No control.** There's no way to cap a runaway script, route cheap requests to cheap models, or cache repeated prompts.

An AI gateway centralises this: services call one API, and the gateway handles authentication, quotas, routing, fallback, caching and metering. This is real AI-platform infrastructure work, and it's what platform teams (e.g. Razorpay's AI platform role) build.

## 2. What this proves to an employer
| Skill | Target job requirement |
|---|---|
| AI infrastructure, provider abstraction, routing | Razorpay (AI platform, infrastructure), Microsoft (AI infrastructure) |
| Reliability: retries, timeouts, circuit breakers, fallback | Amazon/Atlassian (distributed systems), Google (production engineering) |
| Streaming proxy (SSE), async Python at concurrency | Zeko, EaseOps (async processing) |
| Redis rate limiting, quotas, caching | EaseOps (Redis) |
| Observability: Prometheus, OpenTelemetry, Grafana | Razorpay (observability, Prometheus) |
| LLM cost/latency optimisation | EaseOps (LLM cost/performance) |
| API keys, multi-tenant auth | EaseOps (authentication) |

## 3. Scope
### In scope (v1)
- A **unified API** that's OpenAI-compatible (`POST /v1/chat/completions`, `POST /v1/embeddings`), so existing SDKs work by changing `base_url`.
- **Provider adapters:** OpenAI, Anthropic, Google Gemini, and Ollama (local). They translate the request and response formats, including tool/function calls and streaming chunks.
- **Model aliases:** clients ask for `chat-default`, `chat-cheap` or `embed-default`, and config maps an alias to an ordered list of provider:model targets.
- **Routing policies:** priority/failover, weighted, cost-aware (cheapest that meets the requirements), latency-aware (EWMA), and header overrides.
- **Fallback policies:** triggered on timeout, 5xx, 429, provider circuit open, or a context-length error. Non-retryable errors (400 validation, content refusal) are not retried by default.
- Retries with exponential backoff and jitter; per-attempt and overall deadlines; a **circuit breaker** per provider.
- **Streaming passthrough (SSE):** provider streams normalised into OpenAI-style chunks, with first-token latency measured. Mid-stream failure handled per the rule in section 8.
- **Caching:**
  - exact cache: key = hash of the normalised request, for deterministic requests (temperature 0 or when explicitly enabled);
  - optional semantic cache: embedding similarity above a threshold, per tenant, opt-in.
- **API keys:** hashed at rest, scoped to a tenant/project, with allowed models, expiry and revocation.
- **Rate limits and quotas:** requests/min and tokens/min per key (Redis token bucket or sliding window); monthly token and cost budgets per project with a hard or soft cap.
- **Metering:** tokens (prompt/completion/cached), estimated cost from a versioned price table, latency, errors, fallback events, all per request.
- **Observability:** Prometheus metrics, OpenTelemetry traces, a Grafana dashboard, and an admin usage API.
- **Admin API:** manage tenants, projects, keys, routes and the price table (config in YAML plus DB overrides).

### Out of scope (explicitly)
- Prompt management/versioning UI.
- Guardrails and content moderation (a pre/post plugin hook is defined but not implemented).
- Fine-tuning endpoints, image/audio APIs (v1 is chat + embeddings).
- Billing and invoicing (metering only).
- A multi-region deployment.

## 4. Architecture
```
Client (any OpenAI SDK, base_url=gateway)
   │  Authorization: Bearer gk_live_…
   ▼
┌──────────────────────────── FastAPI (uvicorn, async) ────────────────────────────┐
│ 1 AuthN: API key lookup (Redis cache → Postgres)                                  │
│ 2 Quota & rate limit check (Redis: RPM, TPM estimate, monthly budget)             │
│ 3 Request normalisation → internal ChatRequest                                    │
│ 4 Cache lookup (exact → semantic)            ── hit ──▶ response (+ metering)     │
│ 5 Router: alias → candidate targets → policy orders them (health, cost, latency)  │
│ 6 Executor: for target in plan: circuit ok? → adapter.call(deadline) → retry/fallback │
│ 7 Adapter: provider-specific HTTP (httpx pooled clients) + stream normalisation   │
│ 8 Response: normalise → stream/return; post-hook                                   │
│ 9 Metering: usage event → Redis stream → usage worker → Postgres (aggregates)     │
└───────────────────────────────────────────────────────────────────────────────────┘
        │ metrics/traces                    │ usage events
        ▼                                   ▼
  Prometheus / OTel collector ─▶ Grafana    Usage worker ─▶ PostgreSQL (usage_events, daily rollups)
Providers: OpenAI | Anthropic | Gemini | Ollama (local)
```
Why each component exists:
- **The OpenAI-compatible surface:** it's the de-facto standard. Adoption costs one config line, so no client rewrite is needed.
- **Normalised internal model:** adapters translate to and from a single `ChatRequest` / `ChatResponse` / `StreamChunk`. Routing, caching and metering never see provider-specific formats.
- **Router separate from executor:** the router decides *order* (a pure function, easy to test); the executor handles *attempting* (timeouts, retries, breakers). This makes policies swappable and unit-testable.
- **Circuit breaker:** when a provider is failing, stop sending it traffic for a cool-down. This protects latency and avoids piling retries onto a struggling provider.
- **Redis:** handles fast-path state (rate-limit counters, key cache, exact cache, breaker state shared across replicas) and the usage event stream.
- **Asynchronous metering:** writing a usage row synchronously would add DB latency to every call. Instead the request publishes an event and a worker batches inserts. The trade-off: during a crash, a few events may be delayed, but not lost, since Redis Streams hold pending entries until acknowledged.
- **Postgres:** the source of truth for tenants, keys, routes, prices and usage rollups.

## 5. Tech stack & justification
| Choice | Why | Alternatives |
|---|---|---|
| Python 3.12 + FastAPI + uvicorn | target stack; async I/O-bound proxy | Go (faster proxy; Chaitanya knows Go from Ponticare. Choosing Python here fits the target jobs, and the doc explains the performance trade-off) |
| httpx.AsyncClient with pooled connections per provider | HTTP/2 + streaming, timeouts | aiohttp |
| Official provider SDKs **not** used in adapters | shows understanding of the raw APIs and gives full control of streaming/errors | SDKs (convenient, but they hide retries and differ in async support) |
| Redis 7 | rate limits (Lua scripts for atomicity), cache, breaker state, usage stream | in-memory (breaks with multiple replicas) |
| PostgreSQL + SQLAlchemy async + Alembic | config and usage | ClickHouse for usage at huge scale (noted as a future option) |
| prometheus-client, OpenTelemetry SDK | standard metrics and traces | vendor APM |
| Ollama | a free local model for tests, demos and the "final fallback" | vLLM (heavier) |
| pytest + respx (httpx mocking) + testcontainers | deterministic adapter tests | live-only |

## 6. Data model
```
tenants(id pk, name, created_at)
projects(id pk, tenant_id fk, name, monthly_token_budget bigint null, monthly_cost_budget_usd numeric null,
         budget_mode enum[hard,soft], UNIQUE(tenant_id,name))
api_keys(id pk, project_id fk, prefix char(12) unique, key_hash bytea /*sha256+pepper*/, name, allowed_aliases text[] null,
         rpm_limit int, tpm_limit int, expires_at null, revoked_at null, last_used_at, created_at)
providers(name pk /*openai,anthropic,gemini,ollama*/, base_url, enabled, timeout_ms, max_retries)
provider_credentials(provider fk, secret_ref /*env var or secret manager path, never the secret*/)
model_aliases(alias pk, description)
route_targets(alias fk, provider fk, model, priority int, weight int, max_context int, supports_tools bool, supports_stream bool,
              PRIMARY KEY(alias, provider, model))
routing_policies(alias fk pk, policy enum[priority,weighted,cost,latency], config jsonb)
model_prices(provider, model, effective_from date, input_per_mtok numeric, output_per_mtok numeric, cached_input_per_mtok numeric null,
             PRIMARY KEY(provider, model, effective_from))
usage_events(id bigserial, request_id uuid, project_id, api_key_id, alias, provider, model, attempt_count, fallback_used bool,
             cache enum[miss,exact,semantic], stream bool, prompt_tokens, completion_tokens, cached_tokens,
             est_cost_usd numeric(12,6), latency_ms, ttft_ms null, status_code, error_type null, created_at)
             -- partitioned by month
usage_daily(project_id, day, provider, model, requests, prompt_tokens, completion_tokens, est_cost_usd, errors, PK(project_id,day,provider,model))
```
The price table is versioned by `effective_from`, so historical costs stay correct when prices change. Prices are **config values, entered by hand from provider pricing pages, with the source URL and date recorded**. The gateway reports *estimated* cost, clearly labelled as such.

## 7. API / interface design
```
# Data plane (OpenAI-compatible)
POST /v1/chat/completions      {model: "chat-default" | "openai:<model>", messages, tools?, stream?, temperature?, max_tokens?}
     headers (optional): X-Gateway-Route-Policy: cost|latency|priority
                         X-Gateway-Cache: bypass|exact|semantic
                         X-Gateway-Timeout-Ms: 20000
     response headers:   X-Gateway-Provider, X-Gateway-Model, X-Gateway-Attempts, X-Gateway-Cache, X-Request-Id
POST /v1/embeddings
GET  /v1/models                → aliases + concrete models visible to this key

# Control plane (admin JWT)
POST/GET/DELETE /admin/v1/projects, /admin/v1/keys (returns the plaintext key ONCE), /admin/v1/routes, /admin/v1/prices
GET  /admin/v1/usage?project_id=&from=&to=&group_by=model|day|key
GET  /admin/v1/providers/health → breaker state, EWMA latency, error rate

GET /health  /ready  /metrics
```
Routing config example (`config/routes.yaml`, model names are examples only):
```yaml
aliases:
  chat-default:
    policy: priority
    targets:
      - {provider: gemini,    model: "<gemini-flash-model>",  priority: 1}
      - {provider: openai,    model: "<openai-small-model>",  priority: 2}
      - {provider: ollama,    model: "<local-model>",         priority: 3}
    fallback_on: [timeout, http_5xx, http_429, circuit_open]
    retry: {max_attempts_per_target: 2, backoff_ms: [200, 800], jitter: true}
    deadline_ms: 30000
  chat-cheap:
    policy: cost
    requirements: {supports_tools: false, min_context: 16000}
```

## 8. Key engineering problems
1. **Format translation, including tool calls.** Anthropic uses `tool_use`/`tool_result` content blocks, Gemini uses `functionCall`/`functionResponse`, and OpenAI uses `tool_calls`. The adapters map these to the internal format and back. Unsupported features (e.g. tools on a local model that lacks them) make that target ineligible during routing.
2. **Streaming normalisation and mid-stream failure.** Once bytes are sent to the client, falling back to another provider would produce a spliced, inconsistent answer. **Rule:** fallback is only allowed *before the first chunk is sent*. After that, the gateway emits a terminal error event and records a `stream_interrupted` metric. Time-to-first-token gets a separate timeout.
3. **Retry storms.** Retries multiply load during an incident. Mitigations: bounded attempts, exponential backoff with full jitter, an overall request deadline (propagated to adapters as the remaining budget), the circuit breaker, and respecting `Retry-After` on 429s.
4. **Circuit breaker across replicas.** Closed → open when the error rate exceeds a threshold over a rolling window (min N requests) → half-open probe after a cool-down. State lives in Redis so all replicas agree. Race-safe updates use Lua scripts.
5. **Rate limiting tokens before you know the count.** TPM limits need a token count up front. Estimate prompt tokens (tokenizer or chars/4) plus `max_tokens`, reserve that amount, then reconcile with actual usage afterwards (refund or charge the difference). The reserve-and-reconcile approach is atomic in Lua.
6. **Budget enforcement under concurrency.** The monthly budget is checked against a Redis counter, incremented at reconciliation. Hard mode rejects once exceeded; the small overshoot possible from concurrent in-flight requests is documented.
7. **Cache correctness.**
   - The exact-cache key is `sha256(canonical JSON of model alias + resolved target + messages + params)`, stored per tenant (never cross-tenant).
   - Only safe when temperature is 0 or the client opts in; no caching of tool-calling turns by default.
   - The semantic cache is opt-in because of false-hit risk: the threshold is tuned, and hits are logged with their similarity score.
8. **Cost accuracy.** Token counts come from provider usage fields when present, and are estimated otherwise (flagged `estimated=true`). Cached-input pricing and price versioning are handled.
9. **Latency overhead.** The gateway has to add minimal overhead. Pooled connections, no synchronous DB on the hot path (key cache in Redis/memory with a TTL), and async metering. The overhead is **measured** against direct provider calls using a mock provider.

## 9. Milestones
**M1 — Unified API + one adapter + mock provider.**
- Deliverables: the internal models, an OpenAI-compatible endpoint, a **mock provider** (configurable latency, errors, streaming) for tests and load tests, and a real OpenAI-or-Gemini adapter, non-streaming.
- Accept: the official OpenAI Python SDK works against the gateway with only `base_url` changed.

**M2 — All adapters + streaming.**
- Deliverables: Anthropic, Gemini and Ollama adapters; SSE normalisation; tool-call translation; a TTFT measurement.
- Accept: a contract test suite with recorded fixtures per provider (text, tools, stream, errors).

**M3 — Routing, retries, fallback, circuit breaker.**
- Deliverables: the router (4 policies), executor, deadline propagation, Redis breaker, YAML config, and response headers.
- Accept: fault-injection tests with the mock provider (timeouts, 5xx, 429 + Retry-After, circuit open/half-open), and the mid-stream failure rule.

**M4 — Keys, rate limits, quotas.**
- Deliverables: the Postgres schema, admin API, hashed keys, a Lua token bucket (RPM), TPM reserve/reconcile, and budgets.
- Accept: concurrency tests (100 parallel requests never exceed the limit by more than the documented bound).

**M5 — Metering + observability.**
- Deliverables: usage events via Redis Streams, a batch worker, daily rollups, the usage API, the price table, Prometheus metrics, OTel traces, and a Grafana dashboard.
- Accept: the cost calculation is unit-tested against a fixed price table, and the dashboard renders from a load-test run.

**M6 — Caching.**
- Deliverables: exact cache and semantic cache (embeddings via the gateway itself + pgvector or Redis vector), with hit-rate metrics.
- Accept: tests for cache isolation between tenants and the bypass header.

**M7 — Load testing + hardening + release.**
- Deliverables: k6 or Locust scenarios against the mock provider (overhead, throughput, behaviour under provider failure), Docker, compose (gateway, redis, postgres, prometheus, grafana, ollama), CI, and the README with **measured** results.
- Accept: `docs/benchmarks.md` includes the hardware, method and raw results. No numbers without a script.

## 10. Testing strategy
- **Unit tests:** adapters' translation (golden JSON fixtures), router policies (pure functions), backoff/jitter bounds, the breaker state machine, cost calculator, and cache key canonicalisation.
- **Integration tests:** testcontainers Redis/Postgres covering rate-limit Lua scripts, quotas, key auth, and metering end to end.
- **Fault injection:** the mock provider scripted with a latency distribution, error rate, stalled streams and malformed chunks.
- **Compatibility tests:** official OpenAI SDK (sync, async, streaming) against the gateway in CI.
- **Live smoke tests** (manual or nightly, marked): one call per real provider, with a strict token cap.
- **Load tests:** in M7, not in CI.

## 11. Observability
Prometheus metrics:
- `gateway_requests_total{alias,provider,model,status}`
- `gateway_request_duration_seconds{alias,provider}` (histogram)
- `gateway_ttft_seconds{provider}`
- `gateway_tokens_total{provider,model,type}`
- `gateway_cost_usd_total{project,provider,model}`
- `gateway_fallbacks_total{alias,from_provider,to_provider,reason}`
- `gateway_retries_total{provider,reason}`
- `gateway_circuit_state{provider}`
- `gateway_cache_requests_total{type,result}`
- `gateway_rate_limited_total{key_prefix,limit}`
- `gateway_overhead_seconds`

Also:
- OTel spans: request → route → attempt(n) → provider HTTP. Attributes are provider, model and tokens; **no prompt content by default**.
- Grafana dashboard: traffic, latency p50/p95/p99, error and fallback rate, cost per project per day, and cache hit rate.
- **Shared instrumentation with `oss/aiwatch`:** the token/cost/latency event schema and the price table format are designed so aiwatch (a client-side SDK) and the gateway (server side) emit the same usage-event shape. The price table could later become a shared package. This is a deliberate design link between the two projects.

## 12. Security
- API keys: shown once, stored as a salted/peppered SHA-256 with a prefix for lookup, and a constant-time comparison. Revocation takes effect immediately (cache TTL ≤ 30 s, or pub/sub invalidation).
- Provider credentials: only from environment variables or a secret manager, never in the DB in plaintext, and never logged. Per-provider outbound allow-list.
- Tenant isolation: cache keys, usage, budgets and keys are all tenant-scoped; the admin API sits behind a separate auth plane.
- Prompt/response logging is **off by default**. When enabled, it's per project with redaction and a retention TTL.
- Request limits: max body size, max `max_tokens`, max messages; strict validation of the input schema.
- Timeouts on everything; Slowloris protection via server/reverse-proxy settings.

## 13. Deployment
- `docker compose up` brings up the gateway (N replicas behind nginx), redis, postgres, prometheus, grafana and ollama.
- Stateless gateway, so it scales horizontally. All shared state is in Redis or Postgres.
- Cloud via `cloud-infra-lab` (ECS/EC2 + ElastiCache + RDS), optional and torn down after demos. Other projects (`rag-engine`, `ai-agent-platform`) can point their LLM provider at the gateway, which is the cross-project integration demo.

## 14. Evaluation / measurements to collect (all TBD until measured)
| Measurement | Method | Value |
|---|---|---|
| Gateway overhead p50/p95 vs a direct call | k6 against the mock provider (fixed 200 ms latency), direct vs via the gateway | TBD |
| Max sustained RPS per replica (mock provider) | k6 ramp until p95 > SLO or errors > 1% | TBD |
| Availability under a primary-provider outage | mock primary at 100% 5xx for 60 s; measure the success rate and added latency via fallback | TBD |
| Circuit breaker effect | the same outage with the breaker on vs off: latency and wasted attempts | TBD |
| Exact-cache hit rate on a replayed workload | replay a recorded request mix | TBD |
| Semantic-cache false-hit rate | a hand-labelled set of 100 near-duplicate/different prompt pairs | TBD |
| Cost accuracy | gateway estimate vs provider usage dashboard for a small real run | TBD |
| Rate-limit accuracy under concurrency | 200 parallel requests against a limit of 50/min | TBD |

## 15. Prerequisite learning
- `learning/python/`: async, concurrency, streaming generators, typing/protocols.
- `learning/backend/`: fastapi, redis (Lua, streams), rate limiting, API keys/security, postgres partitioning.
- `learning/distributed-systems/`: retries, backoff, jitter, circuit breakers, idempotency.
- `learning/ai/`: provider APIs, tokens, tool-calling formats, cost-optimization, latency-optimization.
- `learning/cloud/`: prometheus, opentelemetry, grafana.
- Recommended after `production-fastapi` M3.

## 16. Interview talking points
- How a request flows through the gateway, and where each millisecond goes (measured overhead).
- Why fallback isn't allowed after the first streamed token.
- Retry storms, and how deadlines, jitter and circuit breakers prevent them.
- Rate limiting tokens you can't count yet (reserve and reconcile).
- Distributed rate limiting correctness: Lua atomicity, and the documented overshoot bound.
- Cache safety: when exact caching is valid, and the semantic cache's false-hit trade-off.
- Build vs buy: how this compares with LiteLLM, Portkey, OpenRouter and cloud gateways, and when you'd use them instead.
- How you'd scale usage analytics (partitioning → ClickHouse).
- Python vs Go for a proxy. Chaitanya has Go experience from Ponticare and can discuss the trade-off.

## 17. Resume bullet templates
- "Built an OpenAI-compatible LLM gateway (FastAPI, Redis, PostgreSQL) routing across OpenAI, Anthropic, Gemini and local Ollama models, with configurable priority/cost/latency policies, retries with jitter, and Redis-backed circuit breakers."
- "Implemented API-key auth, Lua-based RPM/TPM rate limiting and monthly budgets, and asynchronous usage metering via Redis Streams. The gateway adds [MEASURED] ms p95 overhead at [MEASURED] RPS per replica."
- "Kept [MEASURED]% request success during a simulated primary-provider outage through automatic fallback. Prometheus/Grafana dashboards track tokens, estimated cost, fallbacks and TTFT."

## 18. Open questions / uncertainties
- Provider API formats (especially streaming and tool-call deltas) change over time. Verify against current provider docs (via Context7) at implementation time; don't rely on memory.
- Model names and prices are volatile. Keep them in config with the source and date; never hardcode them in code or claim them in docs.
- Semantic cache storage (pgvector vs Redis vector search): decide at M6.
- Whether to support the Anthropic-native `/v1/messages` surface in addition to OpenAI-compatible. Probably not in v1.
- Overlap with `oss/aiwatch`: keep a clear boundary (aiwatch = client-side SDK/CLI for a developer's own app; gateway = a server-side shared service). Share the event schema and the price table only.
- Python performance ceiling for streaming at high concurrency: measure it in M7 before claiming anything. If it's a bottleneck, document it honestly rather than rewriting in Go.
