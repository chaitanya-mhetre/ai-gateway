.PHONY: check test lint type fmt up down run
check: lint type test
test:
	uv run pytest -q
lint:
	uv run ruff check . && uv run ruff format --check .
type:
	uv run mypy
fmt:
	uv run ruff format . && uv run ruff check --fix .
up:
	docker compose up -d redis postgres
down:
	docker compose down
run:
	uv run uvicorn ai_gateway.app:create_app --factory --reload --port 8080
