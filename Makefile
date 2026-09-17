.PHONY: help

# Override with: make api-run PORT=5000
PORT ?= 8000

help:
	@echo "Available commands:"
	@echo ""
	@echo "Development:"
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
	@echo ""
	@echo "Application:"
	@echo "  make api-run      - Run the API server on http://localhost:$(PORT)"
	@echo "  make api-stop     - Stop the process bound to PORT"
	@echo ""
	@echo "Docker (Qdrant):"
	@echo "  make up           - Start Qdrant and wait until healthy"
	@echo "  make down         - Stop Qdrant (indexed vectors are kept)"
	@echo "  make restart      - Restart Qdrant"
	@echo "  make logs         - Follow Qdrant logs"
	@echo "  make status       - Show Qdrant container status"
	@echo "  make clean-all    - Stop Qdrant and delete its volume (indexed vectors)"
	@echo ""
	@echo "Port utilities (default PORT=8000, override with PORT=<port>):"
	@echo "  make check-port   - Check whether PORT is in use"
	@echo "  make kill-port    - Kill the process using PORT"

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

# ===============================
# ========== APPLICATION ========
# ===============================

.PHONY: api-run
api-run:
	@$(MAKE) --no-print-directory check-port 2>/dev/null || { \
		echo; \
		printf "Port $(PORT) is in use. Kill it and continue? (y/N) "; \
		read -r confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			$(MAKE) --no-print-directory kill-port; \
			$(MAKE) --no-print-directory check-port 2>/dev/null || { \
				echo "Port $(PORT) is still in use after the kill attempt."; \
				exit 1; \
			}; \
		else \
			echo "Aborted. Free port $(PORT) manually or use a different port."; \
			exit 1; \
		fi; \
	}
	@echo "Starting API server on http://localhost:$(PORT) (Ctrl-C to stop)..."
	uv run python -m src.main --port $(PORT)

.PHONY: api-stop
api-stop: kill-port

# ===============================
# ============ DOCKER ===========
# ===============================

.PHONY: up
up:
	@echo "Starting Qdrant..."
	docker compose up -d --wait

.PHONY: down
down:
	@echo "Stopping Qdrant (indexed vectors are kept)..."
	docker compose down

.PHONY: restart
restart: down up

.PHONY: logs
logs:
	docker compose logs -f

.PHONY: status
status:
	docker compose ps

.PHONY: clean-all
clean-all:
	@echo "Warning: deleting the Qdrant container and volume (all indexed vectors)."
	@echo "Rebuild the index afterwards with 'uv run rag-index build'."
	docker compose down -v --remove-orphans

# ===============================
# ========= PORT UTILITIES ======
# ===============================

.PHONY: check-port
check-port:
	@if lsof -i :$(PORT) >/dev/null 2>&1; then \
		echo "Port $(PORT) is in use."; \
		exit 1; \
	else \
		echo "Port $(PORT) is free."; \
		exit 0; \
	fi

.PHONY: kill-port
kill-port:
	@if lsof -i :$(PORT) >/dev/null 2>&1; then \
		PID=$$(lsof -ti :$(PORT)); \
		echo "Killing process $$PID on port $(PORT)..."; \
		kill -9 $$PID; \
		sleep 1; \
		if lsof -i :$(PORT) >/dev/null 2>&1; then \
			echo "Port $(PORT) is still in use after the kill attempt."; \
			exit 1; \
		else \
			echo "Port $(PORT) freed."; \
		fi; \
	else \
		echo "No process using port $(PORT)."; \
	fi
