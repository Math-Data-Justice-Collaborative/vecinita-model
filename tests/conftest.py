"""Shared pytest fixtures."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from vecinita.api.routes import create_app


@pytest.fixture()
def mock_ollama_client():
    """Patch the ``ollama.Client`` used inside ``create_app``."""
    with patch("vecinita.api.routes._ollama") as mock_module:
        client = MagicMock()
        mock_module.Client.return_value = client
        yield client


@pytest.fixture()
def test_client(mock_ollama_client):
    """FastAPI TestClient with Ollama stubbed out."""
    app = create_app(ollama_host="http://localhost:11434")
    return TestClient(app)
