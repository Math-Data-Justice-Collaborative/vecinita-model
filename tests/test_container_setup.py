"""Smoke checks for local container runtime files."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_exists_and_uses_asgi_entrypoint() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.11-slim" in dockerfile
    assert "vecinita.asgi:app" in dockerfile


def test_docker_compose_wires_api_to_ollama() -> None:
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "ollama:" in compose
    assert "api:" in compose
    assert "OLLAMA_HOST: http://ollama:11434" in compose
