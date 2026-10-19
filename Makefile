.PHONY: install lint format typecheck test test-unit test-integration ingest up down clean help kind-up kind-down helm-lint helm-template helm-install eval-gate load-test export-feedback

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
	@echo "  make kind-up         Create local multi-node Kind Kubernetes cluster"
	@echo "  make kind-down       Delete local Kind cluster"
	@echo "  make helm-lint       Lint Sentinel Helm chart"
	@echo "  make helm-template   Dry-run render Sentinel Helm manifests"
	@echo "  make helm-install    Install/upgrade Sentinel via Helm on Kubernetes"
	@echo "  make eval-gate       Run Groundedness Quality Gate evaluation benchmark"
	@echo "  make load-test       Run automated Locust load testing suite"
	@echo "  make docs-dev        Run local Mintlify documentation live preview"
	@echo "  make clean           Remove temporary cache files"

docs-dev:
	npx -y mintlify dev --dir docs

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

kind-up:
	kind create cluster --config deploy/kind/kind-cluster.yaml

kind-down:
	kind delete cluster --name sentinel-cluster

helm-lint:
	helm lint deploy/helm/sentinel

helm-template:
	helm template sentinel deploy/helm/sentinel

helm-install:
	helm upgrade --install sentinel deploy/helm/sentinel --namespace sentinel --create-namespace

eval-gate:
	python scripts/evaluate.py --dataset data/evals/golden_dataset.json --min-groundedness 0.70 --offline

load-test:
	locust -f tests/load/locustfile.py --headless -u 10 -r 2 --run-time 15s --host http://localhost:8000

export-feedback:
	python scripts/export_feedback.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
