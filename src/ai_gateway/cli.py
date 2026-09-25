"""Command-line entry points: `ai-gateway serve | usage-worker | migrate | create-admin`."""

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
    if settings.auto_migrate:
        await db.migrate()
    worker = UsageWorker(make_redis(settings.redis_url), db, consumer=socket.gethostname())
    await worker.run_forever()


async def _migrate(settings: Settings, revision: str) -> None:
    from ai_gateway.db import Database

    db = Database(settings.database_url)
    try:
        before = await db.current_revision()
        await db.migrate(revision)
        after = await db.current_revision()
        logging.getLogger("ai_gateway.migrate").info("schema revision %s -> %s", before, after)
    finally:
        await db.dispose()


async def _create_admin(settings: Settings, email: str, role: str) -> None:
    from ai_gateway.admin_users import Role, create_admin_user
    from ai_gateway.db import Database

    db = Database(settings.database_url)
    try:
        if settings.auto_migrate:
            await db.migrate()
        try:
            user, token = await create_admin_user(
                db, settings.key_pepper, email=email, role=Role(role)
            )
        except ValueError as exc:
            raise SystemExit(f"create-admin: {exc}") from exc
    finally:
        await db.dispose()
    # stdout carries only the token so it can be captured: TOKEN=$(ai-gateway create-admin ...)
    print(token)
    logging.getLogger("ai_gateway.admin").info(
        "created admin id=%s email=%s role=%s (the token above is shown once; store it now)",
        user.id,
        user.email,
        user.role,
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="ai-gateway")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the HTTP gateway")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8080)
    sub.add_parser("usage-worker", help="consume usage events from Redis into Postgres")
    migrate = sub.add_parser("migrate", help="apply database migrations (alembic upgrade)")
    migrate.add_argument("revision", nargs="?", default="head")
    create_admin = sub.add_parser("create-admin", help="create an admin user and print its token")
    create_admin.add_argument("--email", required=True)
    create_admin.add_argument("--role", choices=["owner", "operator", "viewer"], default="owner")
    args = parser.parse_args(argv)
    if args.command == "serve":
        _serve(args.host, args.port)
    elif args.command == "migrate":
        asyncio.run(_migrate(Settings(), args.revision))
    elif args.command == "create-admin":
        asyncio.run(_create_admin(Settings(), args.email, args.role))
    else:
        asyncio.run(_worker(Settings()))


if __name__ == "__main__":  # pragma: no cover
    main()
