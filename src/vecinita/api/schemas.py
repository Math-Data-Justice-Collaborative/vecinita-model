"""Pydantic request and response schemas for the Vecinita Model API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _default_model() -> str:
    from vecinita.config import settings

    return settings.default_model


class Message(BaseModel):
    """A single chat message in Ollama-compatible chat format."""

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="Speaker role for this turn.",
        examples=["user"],
    )
    content: str = Field(
        ...,
        description="Plain-text message body.",
        examples=["What affordable housing resources exist in Oakland?"],
    )


class ChatRequest(BaseModel):
    """Request body for the /chat, /api/chat, /stream, and /api/stream endpoints."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "model": "llama3.1:8b",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Summarize tenant rights in two bullet points."
                            ),
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256,
                }
            ]
        }
    )

    model: str = Field(
        default_factory=_default_model,
        description="Ollama model tag available on the server (see GET /health).",
        examples=["llama3.1:8b"],
    )
    messages: list[Message] = Field(
        ...,
        min_length=1,
        description="Ordered chat turns; must include at least one message.",
    )
    temperature: float | None = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature forwarded to Ollama as ``temperature``.",
        examples=[0.7],
    )
    max_tokens: int | None = Field(
        default=None,
        gt=0,
        description=(
            "Upper bound on generated tokens (mapped to Ollama ``num_predict``)."
        ),
        examples=[256],
    )


class ChatResponse(BaseModel):
    """Response body for the /chat endpoint."""

    model: str
    message: Message
    done: bool = True


class StreamChunk(BaseModel):
    """A single chunk yielded by the /stream endpoint (Server-Sent Events)."""

    model: str
    content: str
    done: bool


class HealthResponse(BaseModel):
    """Runtime health and model discovery for load balancers and humans."""

    status: Literal["ok", "error"] = Field(
        ...,
        description="``ok`` when the Ollama client listed models successfully.",
        examples=["ok"],
    )
    models: list[str] = Field(
        default_factory=list,
        description="Model tags reported by the local Ollama server.",
        examples=[["llama3.1:8b", "llama3.2:latest"]],
    )
