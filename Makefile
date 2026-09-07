.PHONY: help

help:
	@echo "Available commands:"
	@echo ""
	@echo "  make install      - Install all dependencies"
	@echo "  make test         - Run tests (concise output)"
	@echo "  make test-verbose - Run tests with verbose output and coverage"
	@echo "  make test-cov     - Run tests with coverage"
	@echo "  make lint         - Run linter (ruff)"
	@echo "  make format       - Format code (ruff)"
	@echo "  make typecheck    - Run type checker (ty)"
	@echo "  make check        - Run all checks (lint + typecheck + test)"
	@echo "  make fetch-data   - Fetch raw eval data (GitHub + Stack Overflow)"
	@echo "  make clean-cache  - Clean up cache and temporary files"

.PHONY: install
install:
	@echo "Installing dependencies..."
	uv sync

.PHONY: test
test:
	@echo "Running tests..."
	uv run pytest -q

.PHONY: test-verbose
test-verbose:
	@echo "Running tests (verbose with coverage)..."
	uv run pytest -vv --cov=src --cov-report=term-missing

.PHONY: test-cov
test-cov:
	@echo "Running tests with coverage..."
	uv run pytest --cov=src --cov-report=term-missing

.PHONY: lint
lint:
	@echo "Running linter..."
	uv run ruff check .

.PHONY: format
format:
	@echo "Formatting code..."
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: typecheck
typecheck:
	@echo "Running type checker..."
	uv run ty check

.PHONY: check
check: lint typecheck test
	@echo "All checks passed."

.PHONY: fetch-data
fetch-data:
	@echo "Fetching eval data..."
	uv run -m scripts.fetch_eval_data github
	uv run -m scripts.fetch_eval_data stackoverflow

.PHONY: clean-cache
clean-cache:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup complete."
