"""Unit tests for Modal app helpers and setup code."""

from __future__ import annotations

import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vecinita import app as app_module
from vecinita.volumes import MODELS_PATH


class TestOllamaEnv:
    def test_sets_models_path_without_mutating_process_env(self):
        original = os.environ.get("OLLAMA_MODELS")
        env = app_module._ollama_env()

        assert env["OLLAMA_MODELS"] == MODELS_PATH
        assert os.environ.get("OLLAMA_MODELS") == original


class TestDownloadModel:
    def test_rejects_unknown_model(self):
        raw_download_model = app_module.download_model.get_raw_f()

        with pytest.raises(ValueError, match="Unknown model 'bad-model'"):
            raw_download_model("bad-model")

    def test_pulls_model_and_commits_volume(self, monkeypatch):
        raw_download_model = app_module.download_model.get_raw_f()
        proc = MagicMock(spec=subprocess.Popen)
        client = MagicMock()
        client.list.return_value = SimpleNamespace(models=[])
        ollama_module = SimpleNamespace(
            Client=MagicMock(return_value=client),
            pull=MagicMock(),
        )
        sleep = MagicMock()
        commit = MagicMock()

        monkeypatch.setattr(app_module, "models_volume", SimpleNamespace(commit=commit))
        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr("time.sleep", sleep)
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)

        raw_download_model("llama3.2")

        subprocess.Popen.assert_called_once()
        ollama_module.pull.assert_called_once_with("llama3.2")
        commit.assert_called_once_with()
        proc.terminate.assert_called_once_with()
        sleep.assert_not_called()

    def test_terminates_server_when_pull_fails(self, monkeypatch):
        raw_download_model = app_module.download_model.get_raw_f()
        proc = MagicMock(spec=subprocess.Popen)
        client = MagicMock()
        client.list.return_value = SimpleNamespace(models=[])
        ollama_module = SimpleNamespace(
            Client=MagicMock(return_value=client),
            pull=MagicMock(side_effect=RuntimeError("pull failed")),
        )

        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)

        with pytest.raises(RuntimeError, match="pull failed"):
            raw_download_model("llama3.2")

        proc.terminate.assert_called_once_with()


class TestApiFactory:
    def test_starts_server_waits_for_readiness_and_builds_fastapi_app(
        self, monkeypatch
    ):
        raw_api = app_module.api.get_raw_f()
        proc = MagicMock(spec=subprocess.Popen)
        client = MagicMock()
        client.list.side_effect = [
            RuntimeError("not ready"),
            SimpleNamespace(models=[]),
        ]
        create_app = MagicMock(return_value="fastapi-app")
        ollama_module = SimpleNamespace(Client=MagicMock(return_value=client))
        sleep = MagicMock()

        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr("time.sleep", sleep)
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)
        monkeypatch.setitem(
            __import__("sys").modules,
            "vecinita.api.routes",
            SimpleNamespace(create_app=create_app),
        )

        result = raw_api()

        assert result == "fastapi-app"
        subprocess.Popen.assert_called_once()
        assert ollama_module.Client.call_count == 2
        sleep.assert_called_once_with(0.5)
        create_app.assert_called_once_with(ollama_host=app_module.settings.ollama_host)
