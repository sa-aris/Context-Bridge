.PHONY: install dev up down run test lint typecheck fmt clean

install:
	uv pip install -e ".[dev]"

dev:
	uv pip install -e ".[dev,fastembed,redis,postgres]"

up:
	docker compose up -d

down:
	docker compose down

run:
	uvicorn context_bridge.api.app:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

test:
	pytest

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests
	ruff format src tests

typecheck:
	mypy src

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.db
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
