.PHONY: install test lint format check docker run

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

check: lint test

docker:
	docker-compose up --build

run:
	uvicorn app.main:app --reload
