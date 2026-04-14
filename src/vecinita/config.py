"""Application configuration and supported model registry."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "vecinita-model"
    default_model: str = "gemma3"
    models_path: str = "/models"
    ollama_host: str = "http://localhost:11434"
    # Seconds to keep a container alive after last request
    scaledown_window: int = 300
    # Request timeout in seconds
    timeout: int = 600


settings = Settings()

# Registry of supported models.
# Each entry maps a friendly model ID to its backend and backend-specific name.
SUPPORTED_MODELS: dict[str, dict] = {
    "gemma3": {"backend": "ollama", "ollama_name": "gemma3"},
    "llama3.2": {"backend": "ollama", "ollama_name": "llama3.2"},
    "llama3.2:1b": {"backend": "ollama", "ollama_name": "llama3.2:1b"},
    "llama3.1": {"backend": "ollama", "ollama_name": "llama3.1"},
    "llama3.1:8b": {"backend": "ollama", "ollama_name": "llama3.1:8b"},
    "mistral": {"backend": "ollama", "ollama_name": "mistral"},
    "phi3": {"backend": "ollama", "ollama_name": "phi3"},
    "gemma2": {"backend": "ollama", "ollama_name": "gemma2"},
    "gemma2:2b": {"backend": "ollama", "ollama_name": "gemma2:2b"},
}
