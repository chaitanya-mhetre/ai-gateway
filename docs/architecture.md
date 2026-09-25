# Architecture

## Request lifecycle (non-streaming)

```
app.chat_completions
 ├─ body-size middleware ............................ 413 if Content-Length > max_body_bytes
 ├─ principal_for → KeyAuthenticator.authenticate .... prefix lookup (TTL cache → DB), HMAC compare, revoked/expired
 ├─ parse_options ................................... X-Gateway-* headers → RequestOptions
 ├─ to_internal ..................................... OpenAI JSON → ChatRequest (validation, max_tokens cap)
 └─ GatewayService.chat
     ├─ Principal.check_alias ....................... per-key allow-list (403)
     ├─ Guard.admit ................................. budget check → RPM take → TPM reserve (429 + Retry-After)
     ├─ ResponseCache.lookup ........................ exact (tenant hash) → semantic (opt-in)   ── hit → refund, record, return
     ├─ Gateway.chat
     │   ├─ resolve ................................. alias or "provider:model"
     │   ├─ router.plan ............................. eligible() then policy ordering (pure)
     │   └─ Executor.chat → _run .................... breaker.allow → attempt(remaining) → retry/backoff → fallback
     ├─ ResponseCache.save
     ├─ PriceTable.cost ............................. Decimal, None if unpriced
     ├─ Guard.settle ................................ TPM reconcile, budget += tokens & micro-USD
     └─ _record ..................................... Prometheus + circuit gauges + UsageRecord → sink
```

## Streaming differences
- `Executor.stream` treats "first chunk received" as the success condition of an attempt. Timeouts
  and errors *before* it trigger retry/fallback like non-streaming calls.
- After the first chunk, `_continue` yields the rest. An error there raises `stream_interrupted`,
  which `app.sse` turns into a `data: {"error": ...}` event, followed by `data: [DONE]`.
- Settlement and usage recording happen in the `finally` of `GatewayService.stream.metered`, so
  they also run when the client disconnects.

## State and where it lives

| State | Single replica (no Redis) | Multi-replica (Redis) |
|---|---|---|
| Circuit breaker | `InMemoryBreaker` | `RedisBreaker` (hash + ZSET, Lua) |
| Rate limits | `InMemoryLimiter` | `RedisLimiter` (hash, Lua token bucket) |
| Budgets | `InMemoryBudgetStore` | `RedisBudgetStore` (HINCRBY, 40-day TTL) |
| Exact cache | `InMemoryCacheStore` (LRU + TTL) | `RedisCacheStore` (SET EX) |
| Semantic cache | in-memory per replica | in-memory per replica (limitation) |
| Usage events | `InProcessSink` → DB | `RedisStreamSink` → `UsageWorker` → DB |
| EWMA latency | per replica | per replica (intentionally local) |
| Keys, tenants, projects, usage | SQLite/Postgres | Postgres |

## Failure modes

| Failure | Behaviour |
|---|---|
| Provider slow | per-attempt timeout = remaining deadline; first-token timeout for streams |
| Provider 5xx / 429 / connection | retry with jitter (Retry-After respected), then fallback; breaker counts it |
| Provider rejects request (400) | no retry, no fallback, 400 to client (another provider would also reject) |
| Context too long for a target | fallback (`context_length`); targets with too-small `max_context` are filtered up front |
| All targets fail | 502 `all_targets_failed` with a per-provider reason summary |
| Redis down | requests fail (limits are fail-closed). A fail-open policy is a possible config (not implemented) |
| Usage worker down | events accumulate in the stream (MAXLEN ~1M); no request impact |
| Worker crash mid-batch | entries stay pending, get re-claimed by XAUTOCLAIM, insert is idempotent |
| DB down | new-key lookups fail (cached keys keep working up to the TTL); usage writes retry |
