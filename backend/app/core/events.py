"""Event contract shared by the voice WebSocket and the frontend client."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel


class AssistantState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    RESPONDING = "responding"


class ServerEvent(BaseModel):
    """Every message the backend pushes over the voice socket."""

    type: Literal[
        "state",
        "wake",
        "partial_transcript",
        "transcript",
        "token",
        "reply_done",
        "audio",
        "audio_done",
        "metrics",
        "error",
    ]
    state: Optional[AssistantState] = None
    text: Optional[str] = None
    # base64-encoded mp3 chunk for `audio` events
    audio: Optional[str] = None
    turn_id: Optional[str] = None
    data: Optional[dict[str, Any]] = None


class ClientEvent(BaseModel):
    type: Literal["config", "text", "interrupt", "start", "stop"]
    text: Optional[str] = None
    data: Optional[dict[str, Any]] = None


class TurnMetrics(BaseModel):
    """Latency breakdown surfaced in the dashboard, all in milliseconds."""

    wake_ms: Optional[float] = None
    stt_ms: Optional[float] = None
    first_token_ms: Optional[float] = None
    first_audio_ms: Optional[float] = None
    total_ms: Optional[float] = None
