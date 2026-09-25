# Benchmarks

All numbers below were **measured** with `bench/overhead.py`. The raw JSON for every run is in
`bench/results/`. Anything not measured is marked TBD.

## Environment
- Laptop: 12th Gen Intel Core i5-1240P, 16 logical cores, Linux 7.0, Python 3.12.14
- Gateway: **one** uvicorn worker (single process), started by the script
- Upstream: `bench/mock_upstream.py`, an OpenAI-compatible fake with a **fixed 200 ms** latency, also one worker
- Load generator: httpx async client on the same machine (it competes for CPU, so these are conservative numbers)
- 2,000 measured requests per run, after 100 warm-up requests; non-streaming chat completions
- `memory` mode: no Redis, auth disabled, usage events written to in-memory SQLite by an in-process task
- `redis` mode: Redis 7 (docker), **auth on** (key lookup + TPM/RPM Lua scripts), usage events sent to a Redis Stream

## Method
The same request goes (a) straight to the upstream and (b) through the gateway to that upstream.
"Overhead" is the difference between (b) and (a) at the same percentile. A difference of
percentiles is not a percentile of differences, so p95 and p99 overhead are indicative only
(that's why p99 can come out lower than p95).

## Results (2026-09-25)

| Mode | Concurrency | Direct p50 / p95 | Via gateway p50 / p95 | Overhead p50 / p95 / mean | Gateway req/s | Errors |
|---|---|---|---|---|---|---|
| memory | 10 | 202.3 / 205.5 ms | 208.3 / 215.6 ms | **6.0 / 10.1 / 6.2 ms** | 47.7 | 0 |
| redis + auth | 10 | 202.9 / 206.9 ms | 215.7 / 225.7 ms | **12.8 / 18.8 / 13.2 ms** | 46.1 | 0 |
| redis + auth | 50 | 202.8 / 217.0 ms | 220.3 / 280.7 ms | 17.5 / 63.7 / 23.3 ms | 215.4 | 0 |
| memory | 50 | 202.2 / 208.7 ms | 216.9 / 367.2 ms | 14.8 / 158.5 / 36.3 ms | 203.8 | 0 |

## What this shows (and doesn't)
- **Uncontended overhead is about 6 ms (in-memory) to 13 ms (Redis + auth) at p50.** Redis mode adds
  about 3–4 round trips per request: the key cache is in-process, but RPM take, TPM take, TPM
  reconcile, breaker record and the usage XADD all go to Redis.
- **At concurrency 50 a single Python process becomes CPU-bound at around 200–215 req/s**, and tail
  latency then grows from queueing inside the gateway. The ceiling for 50 in-flight requests at
  200 ms is 250 req/s.
- The in-memory mode is *slower* under load than Redis mode, because it writes usage rows to SQLite
  inside the same process. That's another reason the production path uses Redis Streams plus a
  separate worker.
- Scaling: the gateway is stateless when Redis is configured, so the next steps are more uvicorn
  workers or replicas. **Not measured yet (TBD).** Profiling the hot path (pydantic re-validation,
  JSON encode/decode twice, per-request Redis round trips that could be pipelined) is the obvious
  optimisation work.

## Provider outage (2026-09-25)

Measured with `bench/outage.py`; per-request raw samples are in `bench/results/outage-*.json` and
the table is printed by `uv run python -m bench.summarise_outage`.

**Setup.** Same laptop as above. Two mock upstreams (`primary`, `secondary`, fixed 200 ms latency)
behind one alias with priority fallback, 1 attempt per target, provider `timeout_ms` 1000, request
deadline 5000 ms. In-memory breaker: window 10 s, min 5 requests, failure ratio 0.5, cool-down 5 s
(`--breaker off` = the circuit never opens). Closed-loop load at concurrency 10 for 60 s:
15 s healthy → 30 s primary failing → 15 s recovered. Single runs on a busy machine, not averages.

Scenarios: **kill** = SIGKILL the primary process (connection refused); **hang** = the primary
accepts the request and never answers; **5xx** = the primary returns 503 immediately.

| Run | Scenario | Breaker | Outage error rate | Failed / tried primary | First fallback | Circuit open | Outage p50 / p95 / max | Primary back |
|---|---|---|---|---|---|---|---|---|
| before fix | hang | off | **100% (60 req)** | 60 / 60 | never | n/a | n/a | 0.06 s |
| before fix | hang | on | 2.3% (939 req) | 22 / 22 | 10.0 s | 10.0 s | 206 / 211 / 266 ms | 5.2 s |
| before fix | kill | off | 0% (1415 req) | 0 / 1415 | 0.22 s | n/a | 210 / 221 / 268 ms | 0.31 s |
| before fix | kill | on | 0% (1424 req) | 0 / 235 | 0.22 s | 5.1 s | 207 / 228 / 270 ms | 5.3 s |
| after fix | hang | off | 0% (250 req) | 0 / 250 | 0.07 s | n/a | 1206 / 1211 / 1255 ms | 0.19 s |
| after fix | hang | on | 0% (1052 req) | 0 / 83 | 0.04 s | 9.7 s | 206 / 1206 / 1249 ms | 2.7 s |
| after fix | kill | on | 0% (1436 req) | 0 / 237 | 0.22 s | 5.0 s | 208 / 215 / 266 ms | 5.2 s |
| after fix | 5xx | on | 0% (1428 req) | 0 / 246 | 0.07 s | 5.2 s | 209 / 217 / 278 ms | 0.39 s |

"Failed / tried primary" = requests that failed / requests that paid for a primary attempt before
falling back. "Circuit open" = first request served by the secondary in a single attempt (primary
skipped). "Primary back" = first request served by the primary after it recovered.

**What this found and what it shows**
- **A real bug.** `ProviderConfig.timeout_ms` was parsed but never applied. A hung primary used the
  whole 5 s request deadline, so with the breaker off *every* request in the outage failed even
  though the secondary was healthy, and throughput collapsed from ~720 to 60 requests per 30 s.
  Fixed in `99f38db` (each non-streaming attempt is capped at `min(timeout_ms, remaining deadline)`);
  covered by `test_hung_provider_is_capped_by_its_timeout_and_falls_back`.
- **After the fix, no request failed in any scenario.** The cost of a hung provider is now one
  provider timeout (~1 s) per request that still tries it.
- **The breaker is what removes that cost.** Hang + breaker off: every request paid ~1.2 s (p50
  1206 ms). Hang + breaker on: only 83 requests paid it, then p50 went back to 206 ms. Kill/5xx fail
  fast anyway, so the breaker mostly saves wasted attempts (237 vs 1415) rather than latency.
- **The breaker takes 5–10 s to open, not instantly.** The failure ratio is computed over a 10 s
  window that still holds the successes from before the outage; it trips only once enough of them
  age out. Hangs are slower (9.7 s) because each failure is only recorded after its 1 s timeout.
  A shorter window or a consecutive-failures trigger would open it faster, at the cost of more
  false trips.
- **Recovery takes up to one cool-down (5 s)** with the breaker on, versus immediate with it off:
  the open circuit keeps skipping the primary until the half-open probe succeeds.

## Not measured yet (TBD)
| Measurement | How |
|---|---|
| Throughput with N workers / replicas | `uvicorn --workers N`, same script |
| Outage runs with Redis-backed breaker and several replicas | same script with `GATEWAY_REDIS_URL` and more gateway processes |
| Repeated outage runs with error bars | several runs per scenario on a quiet machine |
| Exact-cache hit rate on a replayed workload | needs a recorded request mix |
| Semantic-cache false-hit rate | needs a hand-labelled set of prompt pairs |
| Cost accuracy vs provider dashboards | needs a small real-provider run |
| Streaming TTFT overhead | extend the script with SSE |
