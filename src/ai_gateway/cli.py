"""Command-line entry points: `ai-gateway serve` and `ai-gateway usage-worker`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket

from ai_gateway.config import Settings


def _serve(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(
        "ai_gateway.app:create_app",
        factory=True,
        host=host,
        port=port,
        proxy_headers=True,
        log_level="info",
    )


async def _worker(settings: Settings) -> None:
    from ai_gateway.db import Database
    from ai_gateway.metering.usage import UsageWorker
    from ai_gateway.redis_client import make_redis

    if not settings.redis_url:
        raise SystemExit(
            "usage-worker needs GATEWAY_REDIS_URL (without Redis, events are written in-process)"
        )
    db = Database(settings.database_url)
    await db.create_all()
    worker = UsageWorker(make_redis(settings.redis_url), db, consumer=socket.gethostname())
    await worker.run_forever()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="ai-gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    sub.add_parser("usage-worker", help="consume usage events from Redis into Postgres")
    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args.host, args.port)
    else:
        asyncio.run(_worker(Settings()))


if __name__ == "__main__":  # pragma: no cover
    main()
