"""Unit tests for Modal app helpers and setup code."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
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


class TestDeploymentDefaults:
    def test_default_model_is_llama31_8b(self):
        config_source = Path(__file__).resolve().parents[1] / "src/vecinita/config.py"
        content = config_source.read_text(encoding="utf-8")
        assert 'default_model: str = "llama3.1:8b"' in content

    def test_api_function_uses_gpu_acceleration(self):
        app_source_path = Path(__file__).resolve().parents[1] / "src/vecinita/app.py"
        app_source = app_source_path.read_text(encoding="utf-8")
        assert 'gpu="A10G"' in app_source


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

    def test_raises_clear_error_when_server_never_ready(self, monkeypatch):
        raw_download_model = app_module.download_model.get_raw_f()
        proc = MagicMock(spec=subprocess.Popen)
        client = MagicMock()
        client.list.side_effect = RuntimeError("still starting")
        ollama_module = SimpleNamespace(
            Client=MagicMock(return_value=client),
            pull=MagicMock(),
        )

        class FakeTime:
            def __init__(self):
                self.now = 0.0

            def time(self):
                self.now += 31.0
                return self.now

        fake_time = FakeTime()

        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr("time.time", fake_time.time)
        monkeypatch.setattr("time.sleep", MagicMock())
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)

        with pytest.raises(RuntimeError, match="did not become ready"):
            raw_download_model("llama3.2")

        ollama_module.pull.assert_not_called()
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
        ensure_default_model = MagicMock()

        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr("time.sleep", sleep)
        monkeypatch.setattr(
            app_module,
            "_ensure_default_model_downloaded",
            ensure_default_model,
        )
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
        ensure_default_model.assert_called_once_with()
        create_app.assert_called_once_with(ollama_host=app_module.settings.ollama_host)

    def test_fails_fast_and_terminates_when_server_never_ready(self, monkeypatch):
        raw_api = app_module.api.get_raw_f()
        proc = MagicMock(spec=subprocess.Popen)
        client = MagicMock()
        client.list.side_effect = RuntimeError("not ready")
        ollama_module = SimpleNamespace(Client=MagicMock(return_value=client))

        class FakeTime:
            def __init__(self):
                self.now = 0.0

            def time(self):
                self.now += 31.0
                return self.now

        fake_time = FakeTime()
        sleep = MagicMock()
        ensure_default_model = MagicMock()

        monkeypatch.setattr(subprocess, "Popen", MagicMock(return_value=proc))
        monkeypatch.setattr("time.time", fake_time.time)
        monkeypatch.setattr("time.sleep", sleep)
        monkeypatch.setattr(
            app_module,
            "_ensure_default_model_downloaded",
            ensure_default_model,
        )
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)

        with pytest.raises(RuntimeError, match="did not become ready"):
            raw_api()

        proc.terminate.assert_called_once_with()


class TestEnsureDefaultModelDownloaded:
    def test_pulls_and_commits_when_default_model_missing(self, monkeypatch):
        model_id = app_module.settings.default_model
        ollama_name = app_module.SUPPORTED_MODELS[model_id]["ollama_name"]
        client = MagicMock()
        client.list.return_value = SimpleNamespace(
            models=[SimpleNamespace(model="mistral")]
        )
        ollama_module = SimpleNamespace(
            Client=MagicMock(return_value=client),
            pull=MagicMock(),
        )
        commit = MagicMock()

        monkeypatch.setattr(app_module, "models_volume", SimpleNamespace(commit=commit))
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)

        app_module._ensure_default_model_downloaded()

        ollama_module.pull.assert_called_once_with(ollama_name)
        commit.assert_called_once_with()

    def test_skips_pull_when_default_model_present(self, monkeypatch):
        model_id = app_module.settings.default_model
        ollama_name = app_module.SUPPORTED_MODELS[model_id]["ollama_name"]
        client = MagicMock()
        client.list.return_value = SimpleNamespace(
            models=[SimpleNamespace(model=ollama_name)]
        )
        ollama_module = SimpleNamespace(
            Client=MagicMock(return_value=client),
            pull=MagicMock(),
        )
        commit = MagicMock()

        monkeypatch.setattr(app_module, "models_volume", SimpleNamespace(commit=commit))
        monkeypatch.setitem(__import__("sys").modules, "ollama", ollama_module)

        app_module._ensure_default_model_downloaded()

        ollama_module.pull.assert_not_called()
        commit.assert_not_called()
