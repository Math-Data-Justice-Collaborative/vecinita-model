.PHONY: lint test docker-up docker-down docker-logs docker-pull-model deploy verify-health

PYTHONWARNINGS ?= ignore:::requests

lint:
	ruff check .

test:
	PYTHONWARNINGS="$(PYTHONWARNINGS)" pytest

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api

docker-pull-model:
	docker compose exec ollama ollama pull gemma3

deploy:
	PYTHONPATH=src python3 -m modal deploy src/vecinita/app.py

verify-health:
	curl -fsS "$$VECINITA_API_URL/health"