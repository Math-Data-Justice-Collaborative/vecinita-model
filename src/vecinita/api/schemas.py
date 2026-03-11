"""Pydantic request and response schemas for the Vecinita Model API."""

from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single chat message."""

    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request body for the /chat and /stream endpoints."""

    model: str = Field(
        default_factory=lambda: __import__('vecinita.config', fromlist=['settings']).settings.default_model,
        description="Ollama model identifier"
    )
    messages: list[Message] = Field(
        ..., min_length=1, description="Conversation history"
    )
    temperature: float | None = Field(
        default=0.7, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int | None = Field(
        default=None, gt=0, description="Maximum tokens to generate"
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
    """Response body for the /health endpoint."""

    status: Literal["ok", "error"]
    models: list[str] = Field(default_factory=list)
