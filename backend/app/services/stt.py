"""Speech to text on faster-whisper (CTranslate2) with GPU detection and
quantized CPU fallback."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import WHISPER_MODELS, settings
from .audio import clean

logger = logging.getLogger(__name__)

HALLUCINATIONS = {
    "thank you.",
    "thanks for watching!",
    "you",
    "bye.",
    "。",
    "subs by www.zeoranger.co.uk",
}


@dataclass(slots=True)
class Transcript:
    text: str
    language: str
    duration_ms: float
    no_speech_prob: float


def _resolve_device() -> tuple[str, str]:
    device = settings.whisper_device
    if device == "auto":
        try:
            import torch  # type: ignore[import-not-found]

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute = settings.whisper_compute_type
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


class SpeechRecognizer:
    """Lazily loads one model per accuracy tier and keeps it warm."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._lock = asyncio.Lock()
        self.device, self.compute_type = _resolve_device()

    @property
    def available(self) -> bool:
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    async def load(self, mode: Optional[str] = None) -> object:
        mode = mode or settings.accuracy_mode
        async with self._lock:
            if mode not in self._models:
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]

                name = WHISPER_MODELS[mode]
                logger.info("Loading whisper %s on %s/%s", name, self.device, self.compute_type)
                self._models[mode] = await asyncio.to_thread(
                    WhisperModel, name, device=self.device, compute_type=self.compute_type
                )
            return self._models[mode]

    async def warmup(self) -> None:
        try:
            model = await self.load()
            await asyncio.to_thread(
                lambda: list(model.transcribe(np.zeros(settings.sample_rate, dtype=np.float32))[0])  # type: ignore[attr-defined]
            )
        except Exception as error:  # noqa: BLE001 - warmup is best effort
            logger.warning("Whisper warmup skipped: %s", error)

    async def transcribe(
        self, audio: np.ndarray, mode: Optional[str] = None
    ) -> Optional[Transcript]:
        started = time.perf_counter()
        audio = clean(audio)
        if audio.size < settings.sample_rate * settings.vad_min_speech_ms / 1000:
            return None

        model = await self.load(mode)

        def _run() -> tuple[str, str, float]:
            segments, info = model.transcribe(  # type: ignore[attr-defined]
                audio,
                language="en",
                beam_size=1 if (mode or settings.accuracy_mode) == "fast" else 5,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.6,
            )
            collected = list(segments)
            text = " ".join(seg.text.strip() for seg in collected).strip()
            no_speech = (
                min(seg.no_speech_prob for seg in collected) if collected else 1.0
            )
            return text, info.language, no_speech

        text, language, no_speech = await asyncio.to_thread(_run)
        if not text or text.lower().strip() in HALLUCINATIONS or no_speech > 0.6:
            return None

        return Transcript(
            text=text,
            language=language,
            duration_ms=(time.perf_counter() - started) * 1000,
            no_speech_prob=no_speech,
        )


recognizer = SpeechRecognizer()
