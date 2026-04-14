"""Shared pytest fixtures."""

import os

# ``api`` is omitted when ``VECINITA_MODAL_INCLUDE_WEB_ENDPOINTS=0`` at import time.
os.environ.setdefault("VECINITA_MODAL_INCLUDE_WEB_ENDPOINTS", "1")

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vecinita.api.routes import create_app


@pytest.fixture()
def mock_client():
    """Patch ``ollama.Client`` used inside ``create_app`` and yield the mock client."""
    with patch("vecinita.api.routes._ollama") as mock_module:
        client = MagicMock()
        mock_module.Client.return_value = client
        yield client


@pytest.fixture()
def http(mock_client) -> TestClient:
    """FastAPI TestClient with Ollama stubbed out."""
    app = create_app(ollama_host="http://localhost:11434")
    return TestClient(app, raise_server_exceptions=False)
