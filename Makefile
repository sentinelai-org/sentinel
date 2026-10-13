.PHONY: install lint format typecheck test test-unit test-integration ingest up down clean help

help:
	@echo "Available commands:"
	@echo "  make install         Install project dependencies in virtualenv"
	@echo "  make lint            Run ruff linter checks"
	@echo "  make format          Autoformat code using ruff"
	@echo "  make typecheck       Run mypy type analysis"
	@echo "  make test            Run all pytest suites"
	@echo "  make ingest          Ingest Markdown documentation into Qdrant"
	@echo "  make up              Start core services via Docker Compose"
	@echo "  make down            Stop running containers"
	@echo "  make clean           Remove temporary cache files"

install:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy sentinel tests

test:
	pytest -v

test-unit:
	pytest -v tests/unit

test-integration:
	pytest -v tests/integration

ingest:
	python scripts/ingest.py --docs-dir data/docs

up:
	docker compose up -d --build

down:
	docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
