# Contributing

## Setup
```bash
uv sync            # Python 3.12+, installs dev tools
make up            # Redis (:56382) + Postgres (:55435) via docker compose
make check         # ruff lint/format check, mypy --strict, pytest
```

## Rules
- Every change keeps `make check` green; CI runs the same commands plus a Docker smoke test.
- New provider behaviour needs a fixture-based contract test (`tests/fixtures/`) and, ideally, a live
  smoke test (`-m live`) that you ran manually.
- Never commit keys. Provider secrets are only referenced by env-var name in config.
- Never add performance claims without a script in `bench/` and raw results in `bench/results/`.
- Conventional commits: `feat:`, `fix:`, `test:`, `docs:`, `perf:`, `build:`, `refactor:`.

## Adding a provider
1. Subclass `HttpProvider` in `src/ai_gateway/providers/<name>.py`: implement `chat`, `stream`, and optionally `embed`.
2. Map errors via `classify_http_error` so retry/fallback semantics stay consistent.
3. Register the type in `config.ProviderType` and `registry.build_provider`.
4. Add fixtures + tests mirroring `tests/test_adapter_anthropic.py`.
