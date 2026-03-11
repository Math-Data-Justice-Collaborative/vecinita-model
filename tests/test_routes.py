"""Integration-style tests for the FastAPI route handlers.

All Ollama calls are mocked so these tests run without an Ollama server.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vecinita.api.routes import create_app

# ---------------------------------------------------------------------------
# Helpers to build common mock return values
# ---------------------------------------------------------------------------


def _make_chat_response(content: str = "Hello!") -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


def _make_list_response(models: list[str] | None = None) -> MagicMock:
    listed = MagicMock()
    listed.models = [MagicMock(model=m) for m in (models or ["llama3.2"])]
    return listed


def _make_stream_chunks(tokens: list[str]) -> list[MagicMock]:
    chunks = []
    for i, token in enumerate(tokens):
        chunk = MagicMock()
        chunk.message.content = token
        chunk.done = i == len(tokens) - 1
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_client():
    """Return a MagicMock that stands in for ollama.Client(...)."""
    with patch("vecinita.api.routes._ollama") as mock_module:
        client = MagicMock()
        mock_module.Client.return_value = client
        yield client


@pytest.fixture()
def http(mock_client) -> TestClient:
    app = create_app(ollama_host="http://localhost:11434")
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_returns_ok_when_ollama_healthy(self, http, mock_client):
        mock_client.list.return_value = _make_list_response(["llama3.2", "mistral"])
        resp = http.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "llama3.2" in body["models"]
        assert "mistral" in body["models"]

    def test_returns_error_when_ollama_down(self, http, mock_client):
        mock_client.list.side_effect = ConnectionRefusedError("no server")
        resp = http.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "error"
        assert body["models"] == []


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    def test_successful_chat(self, http, mock_client):
        mock_client.chat.return_value = _make_chat_response("Hi there!")
        payload = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["model"] == "llama3.2"
        assert body["message"]["role"] == "assistant"
        assert body["message"]["content"] == "Hi there!"
        assert body["done"] is True

    def test_chat_passes_temperature(self, http, mock_client):
        mock_client.chat.return_value = _make_chat_response()
        payload = {
            "model": "mistral",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.2,
        }
        http.post("/chat", json=payload)
        _, kwargs = mock_client.chat.call_args
        assert kwargs["options"]["temperature"] == 0.2

    def test_chat_passes_max_tokens(self, http, mock_client):
        mock_client.chat.return_value = _make_chat_response()
        payload = {
            "model": "phi3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 128,
        }
        http.post("/chat", json=payload)
        _, kwargs = mock_client.chat.call_args
        assert kwargs["options"]["num_predict"] == 128

    def test_chat_returns_500_on_backend_error(self, http, mock_client):
        mock_client.chat.side_effect = RuntimeError("GPU OOM")
        payload = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "crash?"}],
        }
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 500

    def test_chat_rejects_empty_messages(self, http, mock_client):
        payload = {"model": "llama3.2", "messages": []}
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 422

    def test_chat_multi_turn(self, http, mock_client):
        mock_client.chat.return_value = _make_chat_response("I'm fine.")
        payload = {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
        resp = http.post("/chat", json=payload)
        assert resp.status_code == 200
        _, kwargs = mock_client.chat.call_args
        assert len(kwargs["messages"]) == 3


# ---------------------------------------------------------------------------
# /stream
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    def test_successful_stream(self, http, mock_client):
        tokens = ["Hello", " world", "!"]
        mock_client.chat.return_value = iter(_make_stream_chunks(tokens))
        payload = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        resp = http.post("/stream", json=payload)
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE lines.
        events = [
            json.loads(line.removeprefix("data: "))
            for line in resp.text.splitlines()
            if line.startswith("data: ")
        ]
        contents = [e["content"] for e in events]
        assert contents == tokens
        # Last chunk must be marked done.
        assert events[-1]["done"] is True

    def test_stream_error_yields_error_event(self, http, mock_client):
        mock_client.chat.side_effect = RuntimeError("stream failed")
        payload = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "crash?"}],
        }
        resp = http.post("/stream", json=payload)
        assert resp.status_code == 200
        lines = [
            line for line in resp.text.splitlines() if line.startswith("data: ")
        ]
        assert len(lines) == 1
        event = json.loads(lines[0].removeprefix("data: "))
        assert "error" in event

    def test_stream_rejects_invalid_request(self, http, mock_client):
        resp = http.post("/stream", json={"model": "x", "messages": []})
        assert resp.status_code == 422
