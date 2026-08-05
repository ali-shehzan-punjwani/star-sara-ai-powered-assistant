"""Streaming text to speech.

Synthesis starts on the first sentence rather than the full reply, so audio
begins playing while Groq is still generating. Sentences are also the natural
place for prosody pauses, which is what makes the delivery sound human.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)

# Split on sentence enders, but only after enough characters that we are not
# synthesizing two-word fragments (which sound clipped and cost a round trip).
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[:;])\s+")
_MIN_CHUNK_CHARS = 60
_MAX_CHUNK_CHARS = 240

EMOTION_PROSODY = {
    "neutral": {"rate": settings.tts_rate, "pitch": settings.tts_pitch},
    "warm": {"rate": "+4%", "pitch": "+2Hz"},
    "urgent": {"rate": "+18%", "pitch": "+4Hz"},
    "calm": {"rate": "-6%", "pitch": "-2Hz"},
}


class SentenceChunker:
    """Accumulates streamed tokens and releases speakable chunks."""

    def __init__(self) -> None:
        self._buffer = ""

    def push(self, token: str) -> list[str]:
        self._buffer += token
        chunks: list[str] = []
        while True:
            match = None
            for candidate in _SENTENCE_END.finditer(self._buffer):
                if candidate.end() >= _MIN_CHUNK_CHARS:
                    match = candidate
                    break
            if match:
                chunks.append(self._buffer[: match.end()].strip())
                self._buffer = self._buffer[match.end() :]
                continue
            if len(self._buffer) >= _MAX_CHUNK_CHARS:
                cut = self._buffer.rfind(" ", 0, _MAX_CHUNK_CHARS)
                cut = cut if cut > _MIN_CHUNK_CHARS else _MAX_CHUNK_CHARS
                chunks.append(self._buffer[:cut].strip())
                self._buffer = self._buffer[cut:]
                continue
            break
        return [c for c in chunks if c]

    def flush(self) -> Optional[str]:
        text, self._buffer = self._buffer.strip(), ""
        return text or None


def _speakable(text: str) -> str:
    """Strip markdown that would otherwise be read out character by character."""
    text = re.sub(r"```.*?```", " code block omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"[*_`#>]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


async def synthesize(text: str, emotion: str = "neutral") -> AsyncIterator[bytes]:
    """Yield mp3 chunks for `text` as they are produced by the TTS engine."""
    text = _speakable(text)
    if not text:
        return

    try:
        import edge_tts  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("edge-tts not installed; skipping audio synthesis")
        return

    prosody = EMOTION_PROSODY.get(emotion, EMOTION_PROSODY["neutral"])
    communicate = edge_tts.Communicate(
        text, settings.tts_voice, rate=prosody["rate"], pitch=prosody["pitch"]
    )
    try:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - never kill a turn over audio
        logger.warning("TTS failed for chunk: %s", error)


async def available() -> bool:
    try:
        import edge_tts  # noqa: F401

        return True
    except ImportError:
        return False
