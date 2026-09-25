# Changelog

## [Unreleased]

### Added
- Alembic migrations (`src/ai_gateway/migrations/`), packaged in the wheel so they run from the Docker
  image. `ai-gateway migrate [revision]` applies them; the app runs `upgrade head` on startup only
  when `GATEWAY_AUTO_MIGRATE=true` (the default, for dev/tests).
- `docker compose` runs a one-shot `migrate` service before the gateway and usage worker start.
- `tests/test_migrations.py`: upgrade from empty, downgrade to base, idempotency, data survives
  step-by-step upgrades, and a model-vs-migration drift check. Runs on SQLite and, in CI, on Postgres.

- Admin users with roles (`owner`, `operator`, `viewer`) and personal tokens (`ga_…`, HMAC-hashed with
  the pepper). Migration `0002` adds `admin_users`. New endpoints: `GET /admin/v1/me`,
  `POST|GET /admin/v1/admins`, `DELETE /admin/v1/admins/{id}` (disable; the last owner is protected).
- `ai-gateway create-admin --email --role` bootstraps the first owner and prints its token once.
- Every admin route declares one permission; `tests/test_admin_rbac.py` checks the full role ×
  permission matrix and fails if a route is added without a permission.
- Mutating admin calls are logged to `ai_gateway.admin.audit` (admin id, email, role, permission, route).

- `bench/outage.py`: provider-outage benchmark (kill / hang / 5xx, breaker on vs off) with per-request
  samples, plus `bench/summarise_outage.py`. Results in `docs/benchmarks.md`.

### Fixed
- `ProviderConfig.timeout_ms` is now enforced per non-streaming attempt (capped by the remaining
  deadline). Before, a hung provider consumed the whole request deadline and fallback never ran.

### Removed
- `Database.create_all()`: the schema is no longer created from the ORM models at startup.
- `GATEWAY_ADMIN_TOKEN` (the single shared admin secret). If it's still set, the app logs a warning and
  ignores it.

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
