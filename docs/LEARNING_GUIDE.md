# Learning guide: ai-gateway

This guide is for studying the codebase until you can explain every design decision in an
interview. Read it next to the code. Budget about 10–12 hours in total, spread over a week.

---

## 1. The big picture (read first)

An LLM gateway is a **reverse proxy with opinions**. It sits between your services and LLM
providers and adds reliability, control and visibility. One request goes through:

```
auth → limits → cache → route → execute (retry/fallback/breaker) → translate → meter
```

Three ideas carry the whole design:
1. **One internal format** (`models.py`). Everything provider-specific is pushed to the edges (adapters).
2. **Decide vs do.** The router *decides* the order (pure function); the executor *does* the
   attempts (stateful: time, retries, breaker).
3. **Nothing slow or fragile on the hot path.** Keys are cached, limits are one Lua call, and usage
   is published to a stream and written later by a worker.

---

## 2. File tour, in reading order

| # | File | What to understand |
|---|---|---|
| 1 | `src/ai_gateway/models.py` | The internal vocabulary: `ChatRequest`, `ChatResponse`, `StreamChunk`, `ToolCallDelta`, `Usage`. |
| 2 | `src/ai_gateway/errors.py` | `ProviderError.reason` / `.retryable`; `classify_http_error` maps HTTP status to semantics. |
| 3 | `src/ai_gateway/openai_compat.py` | Public wire format ↔ internal. `ChunkEncoder` builds OpenAI SSE chunks (role only in the first). |
| 4 | `src/ai_gateway/providers/base.py` | `Provider` Protocol, pooled `httpx.AsyncClient`, total deadline via `asyncio.timeout`, the SSE parser `iter_sse`. |
| 5 | `src/ai_gateway/providers/openai.py` | The simplest adapter: payload builder, response parser, streaming parser. |
| 6 | `src/ai_gateway/providers/anthropic.py` | Hard translation: content blocks, role merging, named stream events, tool index mapping. |
| 7 | `src/ai_gateway/providers/gemini.py`, `ollama.py` | id→name mapping for tool results; NDJSON streaming. |
| 8 | `src/ai_gateway/providers/mock.py` | How deterministic fault injection works (`script` FIFO). Every reliability test uses this. |
| 9 | `src/ai_gateway/config.py` | Aliases, targets, retry/fallback config, breaker/cache config, env settings. |
| 10 | `src/ai_gateway/routing/router.py` | `eligible()` + four policies. Pure and easy to test. |
| 11 | `src/ai_gateway/routing/backoff.py`, `health.py` | Full jitter; EWMA. |
| 12 | `src/ai_gateway/routing/breaker.py` | The state machine, twice: in Python and in Lua. |
| 13 | `src/ai_gateway/routing/executor.py` | **The most important file.** `_run` (deadline, retry, fallback) and `stream` / `_continue`. |
| 14 | `src/ai_gateway/gateway.py` | Glue: resolve → plan → execute; `CallMeta`. |
| 15 | `src/ai_gateway/auth.py` | Key format, HMAC + pepper, constant-time compare, TTL cache, `Principal`. |
| 16 | `src/ai_gateway/limits.py` | Token bucket (Python + Lua), reserve/reconcile, budgets, `Guard`. |
| 17 | `src/ai_gateway/cache/*` | Canonical key, eligibility, LRU/TTL, semantic similarity, facade. |
| 18 | `src/ai_gateway/metering/*` | Token estimation, price table, usage events, Streams worker. |
| 19 | `src/ai_gateway/observability/*` | Metric types, label cardinality, spans. |
| 20 | `src/ai_gateway/service.py` | Orchestration: admit → cache → gateway → cost → settle → record. |
| 21 | `src/ai_gateway/app.py`, `admin.py` | HTTP surface, dependency wiring (Redis vs in-memory), lifespan. |
| 22 | `tests/test_executor.py` | Read the tests as a *specification* of the reliability rules. |

**Exercise after each file:** close it and explain out loud what it does and why it exists.

---

## 3. Concepts, with code pointers

### 3.1 Adapter pattern
*Where:* `providers/*.py`, `Provider` Protocol in `providers/base.py`.
Each adapter converts one external interface into the interface the rest of the system expects.
Adding a provider touches one file plus registration; the router, cache and metering don't change.
`Protocol` (structural typing) means adapters don't inherit from a base class to be accepted.
`HttpProvider` exists only to share plumbing.

### 3.2 Deadlines vs timeouts
*Where:* `Executor._run` (`deadline = clock() + deadline_s`, `remaining = deadline - clock()`),
`HttpProvider._post_json` (`asyncio.timeout(timeout)`).
A *timeout* bounds one operation; a *deadline* bounds the whole request. Passing the remaining
budget to every attempt guarantees retries never push a request past its deadline. httpx timeouts
are per read/connect, which is why `asyncio.timeout` is also used to enforce a *total*.

### 3.3 Retries, exponential backoff, full jitter
*Where:* `routing/backoff.py`, retry branch in `Executor._run`.
Only *retryable* errors retry (timeouts, 5xx, 429, connection). Backoff grows per attempt. **Full
jitter** (`uniform(0, base)`) de-synchronises clients so they don't retry in lockstep. `Retry-After`
is a floor. If the wait would exceed the deadline, the executor moves to the next target instead of sleeping.

### 3.4 Fallback policy
*Where:* `AliasConfig.fallback_on`, end of the target loop in `Executor._run`.
Fallback is a *policy*, not "on any error": a 400 means the request itself is bad, so trying
another provider wastes money and time. `context_length` is a fallback reason because a target
with a bigger context window might succeed.

### 3.5 Circuit breaker
*Where:* `routing/breaker.py`, `breaker.allow` / `record_*` calls in the executor.
CLOSED → OPEN when the failure ratio over a rolling window crosses a threshold (with a minimum
request count, so 1 failure out of 1 doesn't trip it). OPEN → HALF_OPEN after a cool-down, then
exactly **one probe**: success closes, failure reopens. Why Redis + Lua: all replicas must agree, and
check-then-set must be atomic (Lua scripts run atomically in Redis). Note which errors count as
"unhealthy" (`HEALTH_FAILURES`). A 400 means the provider is up.

### 3.6 SSE streaming and the no-splice rule
*Where:* `iter_sse` in `providers/base.py`, `ChunkEncoder`, `Executor.stream` / `_continue`, `app.sse`.
SSE is plain text: `data: ...` lines separated by blank lines. HTTP status and headers are sent
**before** the body, so once the first chunk is out, you can't change the status to 502. Hence:
attempt success = "first chunk received" (fallback still possible before that), and mid-stream
errors become an in-band error event.

### 3.7 Token accounting
*Where:* `metering/tokens.py`, `Usage`, adapters' `parse_usage`.
Provider-reported usage is the truth; estimates (chars/4) are used only to *reserve* capacity up front
or when a provider reports nothing, and they're flagged `estimated=True`. Watch Anthropic: its
`input_tokens` excludes cache reads, so the adapter normalises to the OpenAI convention.

### 3.8 Rate limiting: token bucket, reserve → reconcile
*Where:* `limits.py` (`InMemoryLimiter`, `_TAKE`/`_GIVE` Lua, `Guard`).
Bucket capacity = limit/min, refilling continuously. It allows bursts and bounds the average. TPM
can't be known before the response, so reserve `prompt_estimate + max_tokens` and give back the
difference after. Concurrency proof: `test_concurrent_requests_never_exceed_limit_redis` fires 200
parallel requests against a limit of 50 and exactly 50 pass.

### 3.9 Caching
*Where:* `cache/exact.py`, `cache/semantic.py`, `cache/layer.py`.
Exact: a canonical JSON hash, including the **tenant** (isolation). Valid only for deterministic
requests (temperature 0) or when the client opts in; never for tool calls by default. Semantic:
cosine similarity of embeddings, opt-in, with a high threshold. The risk is false hits, so the
similarity is reported. A cache hit refunds the TPM reservation and costs nothing.

### 3.10 Async metering, Redis Streams, at-least-once
*Where:* `metering/usage.py` (`RedisStreamSink`, `UsageWorker.process_once`, `write_batch`).
XADD on the request path → consumer group → batch insert → XACK. If the worker dies before XACK, the
entries stay in the Pending Entries List and are re-claimed with XAUTOCLAIM after `claim_idle_ms`.
That's **at-least-once** delivery; the unique `request_id` makes inserts idempotent, so rows are
effectively-once. `MAXLEN ~` caps memory if the worker is down.

### 3.11 Prometheus metric types
*Where:* `observability/metrics.py`.
- **Counter**: monotonically increasing (requests, tokens, cost). Use `rate()` / `increase()`.
- **Histogram**: bucketed observations (latency, TTFT). Use `histogram_quantile()`, which works across replicas.
- **Gauge**: a current value that goes up and down (circuit state).

Label **cardinality**: every unique label combination is a separate time series, so labels come
only from config (providers/models/aliases) or bounded values (the key prefix), never from user input.

### 3.12 API key security
*Where:* `auth.py`.
Prefix for lookup, HMAC-SHA256 with a pepper held outside the DB, `hmac.compare_digest` against
timing attacks, the same error for unknown prefix and wrong secret, and the key shown only once. Why not
bcrypt? Keys are 256-bit random (no brute force risk), and a per-request slow hash would cost too much latency.

### 3.13 Tracing
*Where:* `observability/tracing.py`, spans in `service.py` and `executor.py`.
One trace per request, a child span per attempt. With no SDK configured, the OTel API is a no-op
(near-zero cost). Prompt content is never recorded.

---

## 4. Things to try (hands-on)

1. `make up && make run`, then `./examples/quickstart.sh` against `:8080`.
2. Break a provider: in `config/gateway.yaml` set `mock-primary.mock.latency_ms: 40000` and send a
   request with `X-Gateway-Timeout-Ms: 2000`. Watch `x-gateway-provider` and `/metrics`.
3. In a Python shell, script `MockProvider("p", script=[http_5xx()] * 20)` and watch the breaker open (see `test_executor.py`).
4. Run `uv run python bench/overhead.py --concurrency 10` and compare with `docs/benchmarks.md`.
5. `docker compose --profile observability up -d` and open Grafana on :53000.
6. Read `redis-cli -p 56382 XINFO GROUPS gw:usage` while sending traffic.

---

## 5. Interview questions, with answers

**Q1. Walk me through a request.**
Auth (key prefix → cached lookup → HMAC compare) → admission (budget, RPM, TPM reserve via Lua)
→ cache lookup → router orders the eligible targets by policy → executor attempts them with the
remaining deadline, retrying retryable errors with jittered backoff and falling back on configured
reasons, skipping open circuits → adapter translates → cost from the price table → settle (TPM
reconcile, budget) → usage event to a Redis Stream → worker writes Postgres.

**Q2. Why can't you fall back after the first streamed token?**
Status and headers are already sent, and the client has partial text. A second model's answer would
be spliced onto the first, producing an inconsistent response. So fallback happens only before the
first chunk; after that the gateway sends an in-band error event and counts `stream_interrupted`.

**Q3. What's a retry storm and how do you prevent one?**
During an incident every client retries, multiplying load on an already failing dependency.
Prevention: cap attempts, exponential backoff with full jitter, respect `Retry-After`, one overall
deadline shared by all attempts, and a circuit breaker to stop sending traffic.

**Q4. Explain full jitter. Why not a fixed backoff?**
A fixed or plain exponential backoff keeps clients synchronised, so they retry together. Full jitter
picks a random delay in [0, base], which spreads retries out. It's the AWS-recommended approach.

**Q5. How does your circuit breaker work across replicas?**
State is in Redis: a hash (state, opened_at, probe flag) and a ZSET of recent outcomes. `allow` and
`record` are Lua scripts, so check-and-update is atomic. Half-open admits one probe using the probe flag.

**Q6. Why is a 400 not counted as a breaker failure?**
The provider answered correctly; the *request* was bad. Counting it would open the circuit on
healthy providers because of one client's bad input. The executor records it as a success.

**Q7. How do you rate-limit tokens you can't count yet?**
Reserve an upper bound (prompt estimate + max_tokens), then reconcile with the actual usage: refund
the difference, or charge extra if under-estimated (the bucket can go negative).

**Q8. Is your distributed rate limit exact?**
The Lua token bucket is exact per key, because each check-and-decrement is atomic in Redis. Test: 200
concurrent requests, limit 50, exactly 50 admitted. Budgets can overshoot by the in-flight
requests' cost, because they're checked before the call and incremented after it. That's documented.

**Q9. Token bucket vs sliding window vs fixed window?**
A fixed window allows 2x bursts at window edges. A sliding log is exact but stores every request.
A sliding window counter approximates it. A token bucket gives controlled bursts with O(1) state
(tokens + timestamp), which suits per-key API limits.

**Q10. When is caching LLM responses safe?**
When the same input should give the same output: temperature 0 or explicit opt-in, no tool calls
(they depend on the outside world), and always scoped per tenant. Semantic caching is opt-in because
of false hits.

**Q11. How would you store API keys?**
Show once, store HMAC-SHA256 with a server-side pepper, look up by a non-secret prefix, compare in
constant time, and return the same error for unknown vs wrong keys. Support expiry and revocation,
with bounded cache staleness.

**Q12. How do you meter usage without slowing requests down?**
Publish an event to a Redis Stream (a sub-millisecond XADD) and let a consumer-group worker batch
insert into Postgres. At-least-once delivery via the PEL + XAUTOCLAIM, idempotent inserts on request_id.

**Q13. At-least-once vs exactly-once?**
Exactly-once delivery isn't achievable across a network plus a DB without transactions spanning
both. The practical approach is at-least-once delivery plus idempotent processing, which gives
effectively-once results.

**Q14. Which Prometheus metric type for latency, and why?**
A histogram: percentiles can be computed and aggregated across replicas with `histogram_quantile`.
Summaries compute quantiles per instance and can't be aggregated.

**Q15. What is label cardinality, and how did you control it?**
Each unique label set is a separate time series; unbounded labels (user id, prompt) explode memory.
Labels come only from config or bounded values like the key prefix.

**Q16. How do you estimate cost correctly when prices change?**
A versioned price table keyed by `effective_from`: the entry in force on the request date is used.
Unknown prices are reported as unknown, never as zero. Cached input tokens use the cached rate.

**Q17. How did you measure gateway overhead?**
The same load goes directly to a fixed-latency mock upstream and through the gateway to it; I compare
percentiles. Uncontended p50 overhead was ~6 ms in-memory and ~13 ms with Redis + auth. With one
worker at concurrency 50, the process became CPU-bound at ~215 req/s. Numbers and raw JSON are in docs/benchmarks.md.

**Q18. Why Python and not Go for a proxy?**
Python matches the target stack, and the proxy is I/O-bound, so asyncio handles concurrency well.
The measured single-process CPU ceiling shows the cost; the fix is more workers/replicas and
hot-path optimisation first. Go would give more headroom per core. It's a trade-off I measured,
not guessed.

**Q19. What happens if Redis goes down?**
Limits, breaker, cache and metering depend on it, so requests fail (fail-closed). An alternative is
fail-open for limits (degraded but available). That's a product decision; I'd make it configurable
and alert on it.

**Q20. How is the router tested?**
It's a pure function: eligible targets and ordering given a config, features, latency snapshot and
price lookup. The weighted policy is tested statistically with a seeded RNG (5,000 samples,
proportion within bounds).

**Q21. How do you test failure handling deterministically?**
The `MockProvider` takes a script of outcomes (timeout, 5xx, 429 with Retry-After, …) consumed per
call, and the executor gets an injectable `sleep`, so tests assert exact retry delays without waiting.

**Q22. Why normalise to an internal model instead of passing OpenAI JSON through?**
Routing, caching and metering would otherwise have to understand four formats. The internal model
isolates vendor changes to one adapter each.

**Q23. How do you keep tenants isolated?**
Tenant in the exact-cache key, per-tenant semantic indexes, usage and budgets keyed by project, and
per-key alias allow-lists. The admin plane has a separate credential.

**Q24. How would you scale usage analytics to billions of rows?**
Partition `usage_events` by month, keep daily rollups for dashboards, and move raw events to a
column store (ClickHouse) with the same event schema.

**Q25. What would you do next?**
Alembic migrations, the outage benchmark (breaker on vs off), multi-worker numbers, guardrail
hooks, a pgvector-backed semantic cache, and a shared usage schema with the aiwatch SDK.

---

## 6. Prior art to compare against
LiteLLM proxy, Portkey, OpenRouter, Cloudflare AI Gateway, Kong AI Gateway. Be ready to say when
you'd buy one of them instead: most teams should, unless they need custom routing, data
residency, or deep integration with internal auth and billing.
