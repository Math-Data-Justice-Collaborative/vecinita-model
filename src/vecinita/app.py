"""Main Modal application.

Deployment
----------
    modal deploy src/vecinita/app.py

Preloading model weights
------------------------
Run once to pull model weights into the persistent volume:

    modal run src/vecinita/app.py::download_model --model-name llama3.2

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

        modal run src/vecinita/app.py::download_model --model-name llama3.2
    """
    import subprocess

    import ollama

    if model_name not in SUPPORTED_MODELS:
        supported = ", ".join(SUPPORTED_MODELS)
        raise ValueError(
            f"Unknown model '{model_name}'. Supported models: {supported}"
        )

    ollama_name = SUPPORTED_MODELS[model_name]["ollama_name"]

    # Start the Ollama daemon so we can issue pull commands through it.
    proc = subprocess.Popen(
        ["ollama", "serve"],
        env=_ollama_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_ollama_ready(timeout_seconds=30)

        print(f"Pulling {ollama_name} …")
        ollama.pull(ollama_name)

        # Persist changes to the volume.
        models_volume.commit()
        print(f"Successfully downloaded '{ollama_name}' into the models volume.")
    finally:
        proc.terminate()


# ---------------------------------------------------------------------------
# Web API endpoint
# ---------------------------------------------------------------------------


@app.function(
    image=ollama_image,
    volumes={MODELS_PATH: models_volume},
    scaledown_window=settings.scaledown_window,
    timeout=settings.timeout,
    # Remove or change the GPU spec to match your Modal plan / model size.
    # gpu=modal.gpu.A10G(),
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
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
