.PHONY: help install sync lock lint format format-check typecheck test test-unit test-integration test-e2e cov security audit precommit clean serve docker-build compose-up compose-down

UV ?= uv
PYTHON_PACKAGE := transduce
COVERAGE_THRESHOLD := 80

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage: make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install the project and dev dependencies via uv sync
	$(UV) sync

sync: install ## Alias for install

lock: ## Refresh uv.lock without installing
	$(UV) lock

lint: ## Run ruff check (lint)
	$(UV) run ruff check src/ tests/

format: ## Apply ruff format in-place
	$(UV) run ruff format src/ tests/

format-check: ## Verify formatting without modifying files
	$(UV) run ruff format --check src/ tests/

typecheck: ## Run mypy --strict
	$(UV) run mypy --config-file=pyproject.toml

test: test-unit ## Default test target (unit tests)

test-unit: ## Run unit tests
	$(UV) run pytest -m unit

test-integration: ## Run integration tests (requires real backends)
	$(UV) run pytest -m "integration and not slow"

test-e2e: ## Run end-to-end tests (requires docker compose stack)
	$(UV) run pytest -m e2e

cov: ## Run unit tests with coverage and enforce the project threshold
	$(UV) run pytest -m unit \
		--cov=src/$(PYTHON_PACKAGE) \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=$(COVERAGE_THRESHOLD)

security: ## Run bandit security checks
	$(UV) run bandit -c pyproject.toml -r src/

audit: ## Audit installed dependencies for known vulnerabilities
	$(UV) run pip-audit --strict

precommit: ## Run all configured pre-commit hooks against every file
	$(UV) run pre-commit run --all-files

clean: ## Remove build artifacts and caches
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

serve: ## Run the transduce service locally (requires the api/cli implementation)
	$(UV) run python -m $(PYTHON_PACKAGE).cli serve --config transduce.example.yaml

docker-build: ## Build the runtime container image
	docker build -t $(PYTHON_PACKAGE):dev .

compose-up: ## Start the local docker compose stack (transduce + ollama)
	docker compose up -d

compose-down: ## Stop the local docker compose stack
	docker compose down
