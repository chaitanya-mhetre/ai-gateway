# ai-gateway

A self-hosted, **OpenAI-compatible LLM gateway**. It puts OpenAI, Anthropic, Google Gemini and local
Ollama models behind one API, with configurable routing, retries, fallback, circuit breakers,
caching, API keys, rate limits, budgets, and token/cost/latency metering.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:58080/v1", api_key="gk_...")  # the only change
client.chat.completions.create(model="chat-default", messages=[{"role": "user", "content": "hi"}])
```

> Status: portfolio / learning project. It works end to end locally (tests, Docker, compose
> stack), but it has **not** served production traffic. Every performance number here was
> measured with a committed script; everything else is marked TBD.

---

## Problem

Once a company has more than one LLM feature, every team runs into the same problems:
- **Outages and lock-in:** one provider has a bad hour, and every feature built on it breaks.
- **No cost visibility:** "what did AI cost last month, per team?" has no answer.
- **Duplicated plumbing:** every service re-implements retries, timeouts, keys, rate limits and logging.
- **No control:** nobody can cap a runaway script, route cheap work to cheap models, or cache repeated prompts.

## Why it exists

A gateway centralises all of this. Services call one API, and the gateway handles authentication,
quotas, routing, fallback, caching and metering. Similar products exist (LiteLLM, Portkey,
OpenRouter, cloud AI gateways). This project exists to **build and understand** that
infrastructure end to end: raw provider wire formats, reliability patterns, and distributed
rate limiting.

## Architecture

```
Client (any OpenAI SDK, base_url = gateway)        Authorization: Bearer gk_<prefix>_<secret>
   │
   ▼
FastAPI ─ body-size limit ─ auth (HMAC key lookup, TTL cache) ─ admission: budget ▸ RPM ▸ TPM reserve (Redis Lua)
   │
   ├─ cache lookup: exact (tenant-scoped hash) ▸ semantic (opt-in, embedding similarity) ──hit──▶ response
   │
   ▼
Router (pure function)          alias → eligible targets (tools / stream / context) → order by policy
   │                            priority | weighted | cost (price table) | latency (EWMA)
   ▼
Executor                        per target: circuit open? skip ▸ attempt(remaining deadline)
   │                            retry w/ full-jitter backoff + Retry-After ▸ fallback if reason ∈ fallback_on
   │                            streams: fallback only BEFORE the first token
   ▼
Adapters (raw httpx, pooled)    OpenAI │ Anthropic │ Gemini │ Ollama │ Mock
   │
   ▼
Settle: TPM reconcile ▸ budget += tokens/cost ▸ usage event ─▶ Redis Stream ─▶ usage-worker ─▶ Postgres (+ daily rollups)
Observability: Prometheus /metrics · OpenTelemetry spans (request → attempt) · Grafana dashboard
```

More detail: [docs/architecture.md](docs/architecture.md). Provider format notes: [docs/providers.md](docs/providers.md).

## Features

| Area | What's implemented |
|---|---|
| API | `POST /v1/chat/completions` (JSON + SSE streaming, tools), `POST /v1/embeddings`, `GET /v1/models` |
| Providers | OpenAI (and compatible servers), Anthropic Messages, Gemini generateContent, Ollama, and a scriptable mock |
| Translation | system prompts, tool calls ↔ `tool_use` / `functionCall`, tool results, stop reasons, usage, streaming deltas |
| Routing | aliases → ordered targets; `priority`, `weighted`, `cost`, `latency` policies; capability/context filtering; `X-Gateway-Route-Policy` override |
| Reliability | overall deadline (`X-Gateway-Timeout-Ms`), per-target retries with full jitter, `Retry-After` floor, configurable `fallback_on`, circuit breaker (in-memory or Redis Lua, shared across replicas), first-token timeout |
| Streaming | provider streams normalised to OpenAI chunks; mid-stream failure becomes an in-band error event (no splicing) |
| Keys | `gk_<prefix>_<secret>`, stored as HMAC-SHA256 + pepper, constant-time compare, shown once, revocable, expiry, per-key alias allow-list |
| Limits | RPM and TPM token buckets (atomic Lua); TPM reserve → reconcile; monthly token/cost budgets (hard/soft) |
| Metering | versioned price table → estimated cost; usage events through Redis Streams with an at-least-once worker and idempotent inserts; daily rollups; usage API |
| Caching | exact cache (temperature 0 or opt-in, tenant-isolated); semantic cache (opt-in, threshold, similarity reported) |
| Observability | 13 Prometheus metrics, OTel spans (no prompt content), 12-panel Grafana dashboard |
| Admin | tenants, projects, keys, provider health, usage, prices (separate admin token) |

## Tech stack

Python 3.12 · FastAPI · httpx · Pydantic v2 · Redis 7 (Lua, Streams) · PostgreSQL 16 / SQLite ·
SQLAlchemy 2.0 async · prometheus-client · OpenTelemetry · Docker · GitHub Actions · pytest + respx · ruff · mypy --strict.

**Why no vendor SDKs in the adapters?** Full control over timeouts, streaming, and error
classification, and it forces understanding of each wire format.

## Quick start

```bash
# 1. Offline, no keys needed: mock providers
docker compose up -d --build            # gateway :58080, redis :56382, postgres :55435, usage-worker
./examples/quickstart.sh                 # creates tenant → project → key, calls the gateway, shows usage

# 2. With dashboards
docker compose --profile observability up -d   # prometheus :59090, grafana :53000 (dashboard "AI Gateway")

# 3. Local development
uv sync && make up && make check         # lint + mypy --strict + tests (Redis tests run when Redis is up)
make run                                 # uvicorn --reload on :8080 using config/gateway.yaml
```

For real providers, copy `config/gateway.example.yaml`, fill in model names, add the provider keys
to `.env` (see `.env.example`), and set `GATEWAY_CONFIG_PATH`.

## API usage

```bash
curl localhost:58080/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -H 'X-Gateway-Route-Policy: cost' -H 'X-Gateway-Timeout-Ms: 15000' \
  -d '{"model":"chat-default","temperature":0,"messages":[{"role":"user","content":"hi"}]}' -i
# x-gateway-provider: gemini   x-gateway-model: ...   x-gateway-attempts: 2   x-gateway-cache: miss
```

| Request header | Values |
|---|---|
| `X-Gateway-Route-Policy` | `priority` · `weighted` · `cost` · `latency` |
| `X-Gateway-Cache` | `bypass` · `exact` (opt in even at temperature > 0) · `semantic` |
| `X-Gateway-Timeout-Ms` | overall deadline (capped at the alias deadline) |

Admin (`Authorization: Bearer $GATEWAY_ADMIN_TOKEN`): `POST /admin/v1/tenants`, `POST /admin/v1/projects`,
`POST|GET /admin/v1/keys`, `DELETE /admin/v1/keys/{id}`, `GET /admin/v1/usage?project_id=&group_by=model|day|key`,
`GET /admin/v1/providers/health`, `GET /admin/v1/prices`. OpenAPI UI at `/docs`.

## Testing

```bash
make check                 # ruff + mypy --strict + pytest
uv run pytest -m redis     # Lua scripts, Streams worker, shared breaker (needs `make up`)
uv run pytest -m live      # manual: one tiny call per real provider (needs keys + LIVE_*_MODEL env vars)
```
- **Adapters:** contract tests against fixtures per provider (text, tools, streaming, errors).
  The fixtures are hand-written from the documented formats; see `tests/fixtures/README.md`.
- **Fault injection:** a scripted `MockProvider` covers timeouts, 5xx, 429 + Retry-After, breaker
  open/half-open, deadline exhaustion, first-token timeout, and mid-stream failure.
- **Concurrency:** 200 parallel requests against a limit of 50/min admit **exactly** 50 (Redis Lua).
- **Compatibility:** the official `openai` SDK (sync, streaming, embeddings) runs against the app in-process.
- 139 tests, plus 3 manual live tests.

## Deployment

- `docker compose up` runs gateway + usage-worker + Redis + Postgres (+ Prometheus/Grafana, + Ollama via profiles).
- The gateway is stateless when `GATEWAY_REDIS_URL` is set. Breaker state, rate limits, budgets, the cache
  and usage events all live in Redis, so it scales horizontally behind a load balancer.
- Without Redis it runs as a single replica with in-memory state (fine for local dev).
- The image is a multi-stage build and runs as non-root, with a healthcheck.
- Cloud deployment (ECS/EC2 + ElastiCache + RDS) is planned in the separate `cloud-infra-lab` project.

## Security

- API keys are shown once and stored as HMAC-SHA256 with a pepper that's kept outside the DB, compared
  in constant time. Unknown prefix and wrong secret return the same error.
- Revocation is immediate on the replica that revoked the key, and within `key_cache_ttl_s` (30 s) elsewhere.
- Provider credentials are read only from the env vars named in the config. They're never stored in the DB or logged.
- Tenant isolation: cache keys and semantic indexes are tenant-scoped; usage and budgets are per project.
- Prompt/response content is not logged and never goes into spans or metrics labels.
- Input limits: body size, `max_tokens` cap, message count, and at most 4 stop sequences. Text-only content (v1).
- The admin plane uses a separate token (production should use SSO/JWT with roles; see Limitations).

## Performance (measured)

Local laptop, single uvicorn worker, mock upstream with a fixed 200 ms latency. Full method and raw data are in [docs/benchmarks.md](docs/benchmarks.md).

| Mode | Concurrency | Added latency p50 / p95 | Gateway req/s |
|---|---|---|---|
| in-memory, auth off | 10 | 6.0 / 10.1 ms | 47.7 |
| Redis + auth + limits | 10 | 12.8 / 18.8 ms | 46.1 |
| Redis + auth + limits | 50 | 17.5 / 63.7 ms | 215.4 (CPU-bound single process) |

Not yet measured (TBD): multi-worker throughput, availability during a simulated provider outage,
cache hit rates, cost accuracy against provider dashboards.

## Engineering trade-offs

- **Fallback only before the first streamed token.** After that, the client has partial output, and splicing
  a second model's answer would be wrong. The gateway emits an in-band error instead.
- **Router vs executor split.** Ordering is a pure function (easy to unit-test every policy); attempting
  (timeouts, retries, breakers) is stateful and lives separately.
- **Retry storms.** Bounded attempts, full jitter, a `Retry-After` floor, one deadline shared by all
  attempts, and a circuit breaker. Retries never extend a request past its deadline.
- **TPM before you know the count:** reserve an upper bound, then reconcile. The bucket can go negative
  after an under-estimate, which delays that key rather than failing another request.
- **Budgets under concurrency** can overshoot by at most the in-flight requests' cost. Documented, accepted for metering.
- **Async metering:** usage is never a synchronous DB write on the request path. With Redis it's at-least-once
  plus idempotent inserts; the in-process mode can lose queued events on a crash.
- **Exact vs semantic cache:** exact is safe but low-hit. Semantic raises the hit rate but risks false hits,
  so it's opt-in, threshold-gated and reports its similarity.
- **Python vs Go for a proxy:** Python matches the target stack; the measured single-process ceiling (about 200 req/s
  at 50 concurrent) shows the cost. The next step is more workers/replicas and hot-path profiling, not a rewrite.

## Limitations

- No Alembic migrations yet (tables come from `create_all`); no Postgres partitioning of `usage_events`.
- The admin API uses a static bearer token, not SSO/RBAC; price and route edits are config files, not DB-backed.
- The semantic cache is an in-memory brute-force index (per replica). Production would use pgvector or Redis vector search.
- Text-only messages (no images/audio). Anthropic-native `/v1/messages` isn't exposed.
- Gemini support targets `generateContent`; Google's newer Interactions API isn't implemented.
- Provider fixtures are hand-written from the docs; refresh them with `pytest -m live` recordings.

## Roadmap

- Alembic migrations; monthly partitions for `usage_events`
- Outage/breaker benchmark and multi-worker throughput numbers
- Guardrail plugin hooks (pre/post), prompt/response logging with redaction and retention (opt-in)
- pgvector semantic cache; share the usage-event schema and price table with `oss/aiwatch`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under [MIT](LICENSE).
