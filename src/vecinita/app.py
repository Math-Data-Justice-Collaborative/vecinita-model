"""Main Modal application.

Deployment
----------
    modal deploy src/vecinita/app.py

Preloading model weights
------------------------
Run once to pull model weights into the persistent volume:

    modal run src/vecinita/app.py::download_model --model-name llama3.1:8b

Available model IDs are listed in ``config.SUPPORTED_MODELS``.

Local development
-----------------
    modal serve src/vecinita/app.py
"""

from __future__ import annotations

import modal

from vecinita.config import SUPPORTED_MODELS, settings
from vecinita.images import ollama_image
from vecinita.volumes import MODELS_PATH, models_volume

app = modal.App(settings.app_name)

# ---------------------------------------------------------------------------
# Weight pre-loading function
# ---------------------------------------------------------------------------


@app.function(
    image=ollama_image,
    volumes={MODELS_PATH: models_volume},
    timeout=3600,  # allow up to 1 h for large models
)
def download_model(model_name: str) -> None:
    """Pull *model_name* into the shared Modal volume.

    This function should be run once (or whenever you add a new model):

        modal run src/vecinita/app.py::download_model --model-name llama3.1:8b
    """
    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {supported}"
        )

    _download_model_if_missing(model_name)


@app.function(
    image=ollama_image,
    volumes={MODELS_PATH: models_volume},
    timeout=3600,
)
def download_default_model() -> None:
    """Ensure the configured default model exists in the shared volume."""
    _download_model_if_missing(settings.default_model)


# ---------------------------------------------------------------------------
# Web API endpoint
# ---------------------------------------------------------------------------


@app.function(
    image=ollama_image,
    volumes={MODELS_PATH: models_volume},
    scaledown_window=settings.scaledown_window,
    timeout=settings.timeout,
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app(requires_proxy_auth=False)
def api() -> object:
    """Expose the FastAPI application as a Modal web endpoint.

    The Ollama daemon is started once per container and reused across
    requests for the lifetime of the container (``scaledown_window``).
    """
    import subprocess

    # Start the Ollama server.
    proc = subprocess.Popen(  # noqa: F841  (kept to allow clean shutdown if needed)
        ["ollama", "serve"],
        env=_ollama_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait until the server is accepting requests; fail fast if not ready.
    try:
        _wait_for_ollama_ready(timeout_seconds=30)
    except RuntimeError:
        proc.terminate()
        raise

    _ensure_default_model_downloaded()

    from vecinita.api.routes import create_app

    return create_app(ollama_host=settings.ollama_host)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ollama_env() -> dict[str, str]:
    """Return an environment mapping with OLLAMA_MODELS set."""
    import os

    env = os.environ.copy()
    env["OLLAMA_MODELS"] = MODELS_PATH
    return env


def _wait_for_ollama_ready(timeout_seconds: int = 30) -> None:
    """Wait until Ollama responds or raise a clear timeout error."""
    import time

    import ollama

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            ollama.Client().list()
            return
        except Exception:
            time.sleep(0.5)

    raise RuntimeError(
        f"Ollama server did not become ready within {timeout_seconds} seconds."
    )


def _ensure_default_model_downloaded() -> None:
    """Pull the configured default model if it is missing in the volume."""
    import ollama

    model_id = settings.default_model
    metadata = SUPPORTED_MODELS.get(model_id)
    if metadata is None:
        raise RuntimeError(
            f"Default model '{model_id}' is not present in SUPPORTED_MODELS."
        )

    ollama_name = metadata["ollama_name"]
    listed = ollama.Client().list()
    installed = {m.model for m in listed.models}
    if ollama_name in installed:
        return

    print(
        f"Default model '{ollama_name}' not found in volume. Pulling model now..."
    )
    ollama.pull(ollama_name)
    models_volume.commit()
    print(f"Default model '{ollama_name}' is ready.")


def _download_model_if_missing(model_name: str) -> None:
    """Pull *model_name* into the shared volume only when missing."""
    import subprocess

    import ollama

    ollama_name = SUPPORTED_MODELS[model_name]["ollama_name"]

    # Start the Ollama daemon so we can issue pull/list commands through it.
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=_ollama_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_ollama_ready(timeout_seconds=30)
        installed = {m.model for m in ollama.Client().list().models}
        if ollama_name in installed:
            print(f"Model '{ollama_name}' already present in volume; skipping pull.")
            return

        print(f"Pulling {ollama_name} ...")
        ollama.pull(ollama_name)

        # Persist changes to the volume.
        models_volume.commit()
        print(f"Successfully downloaded '{ollama_name}' into the models volume.")
    finally:
        proc.terminate()
