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

## Not measured yet (TBD)
| Measurement | How |
|---|---|
| Throughput with N workers / replicas | `uvicorn --workers N`, same script |
| Availability under a primary-provider outage | fault-inject the mock upstream (100% 5xx for 60 s) with a second upstream as fallback |
| Circuit breaker on vs off during an outage | same scenario; compare latency and wasted attempts |
| Exact-cache hit rate on a replayed workload | needs a recorded request mix |
| Semantic-cache false-hit rate | needs a hand-labelled set of prompt pairs |
| Cost accuracy vs provider dashboards | needs a small real-provider run |
| Streaming TTFT overhead | extend the script with SSE |
