.PHONY: install test lint format fix check docker run

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=ai --cov=app --cov-report=term-missing

lint:
	ruff check .
	pyright .

format:
	ruff format .
	ruff check --fix .

fix:
	-ruff check . --fix --unsafe-fixes
	@echo Auto-fix complete. Run 'make lint' to review remaining issues.

check: lint test

docker:
	docker-compose up --build

run:
	uvicorn app.main:app --reload
