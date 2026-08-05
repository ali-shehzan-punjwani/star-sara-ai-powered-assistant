"""Wake word detection.

Primary backend is openWakeWord (small CNN over mel frames, ~1 % of a core and
typically <300 ms from utterance to activation). Porcupine is used when an
access key is configured. When neither is installed the pipeline falls back to
fuzzy matching the Whisper transcript, which is slower but always available.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional, Protocol

import numpy as np

from ..config import settings
from ..core.store import similarity
from .audio import float32_to_pcm16

logger = logging.getLogger(__name__)

# Stand-ins until a "STAR SARA" model is trained: both are two-syllable names
# with the same stress pattern, which keeps false triggers low.
DEFAULT_WAKE_MODELS = ["hey_jarvis", "alexa"]


class WakeBackend(Protocol):
    name: str

    def process(self, frame: np.ndarray) -> bool: ...


class OpenWakeWordBackend:
    name = "openwakeword"

    def __init__(self) -> None:
        from openwakeword.model import Model  # type: ignore[import-not-found]

        # A custom "star sara" model is used when OPENWAKEWORD_MODEL_PATH points
        # at one; otherwise the bundled phrases stand in. ONNX runs everywhere,
        # unlike the tflite runtime.
        custom = os.getenv("OPENWAKEWORD_MODEL_PATH")
        self._model = Model(
            wakeword_models=[custom] if custom else DEFAULT_WAKE_MODELS,
            inference_framework="onnx",
        )
        self._threshold = float(os.getenv("OPENWAKEWORD_THRESHOLD", "0.5"))

    def process(self, frame: np.ndarray) -> bool:
        pcm = np.frombuffer(float32_to_pcm16(frame), dtype=np.int16)
        scores = self._model.predict(pcm)
        return any(score >= self._threshold for score in scores.values())


class PorcupineBackend:
    name = "porcupine"

    def __init__(self) -> None:
        import pvporcupine  # type: ignore[import-not-found]

        key = os.environ["PORCUPINE_ACCESS_KEY"]
        keyword_paths = os.getenv("PORCUPINE_KEYWORD_PATHS")
        if keyword_paths:
            self._engine = pvporcupine.create(
                access_key=key, keyword_paths=keyword_paths.split(",")
            )
        else:
            self._engine = pvporcupine.create(access_key=key, keywords=["jarvis", "computer"])
        self._frame_length = self._engine.frame_length
        self._buffer = np.zeros(0, dtype=np.int16)

    def process(self, frame: np.ndarray) -> bool:
        pcm = np.frombuffer(float32_to_pcm16(frame), dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, pcm])
        detected = False
        while self._buffer.size >= self._frame_length:
            chunk, self._buffer = (
                self._buffer[: self._frame_length],
                self._buffer[self._frame_length :],
            )
            if self._engine.process(chunk) >= 0:
                detected = True
        return detected


def _build_backend() -> Optional[WakeBackend]:
    if os.getenv("PORCUPINE_ACCESS_KEY"):
        try:
            backend = PorcupineBackend()
            logger.info("Wake word backend: %s", backend.name)
            return backend
        except Exception as error:  # noqa: BLE001
            logger.warning("Porcupine unavailable: %s", error)
    # The bundled openWakeWord phrases are not "STAR SARA", so the acoustic
    # engine is only trusted when a custom model is supplied (or explicitly
    # opted into). Otherwise transcript matching keeps the wake word correct.
    if os.getenv("OPENWAKEWORD_MODEL_PATH") or os.getenv("OPENWAKEWORD_USE_BUNDLED") == "1":
        try:
            backend = OpenWakeWordBackend()
            logger.info("Wake word backend: %s", backend.name)
            return backend
        except Exception as error:  # noqa: BLE001
            logger.warning("openWakeWord unavailable: %s", error)
    logger.info("Wake word backend: transcript fuzzy matching")
    return None


class WakeWordDetector:
    def __init__(self) -> None:
        self._backend = _build_backend()
        self._last_trigger = 0.0

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "transcript-fuzzy"

    @property
    def acoustic(self) -> bool:
        return self._backend is not None

    def _cooled_down(self) -> bool:
        now = time.monotonic()
        if now - self._last_trigger < settings.wake_word_cooldown_seconds:
            return False
        self._last_trigger = now
        return True

    def process_frame(self, frame: np.ndarray) -> bool:
        if not self._backend:
            return False
        return bool(self._backend.process(frame)) and self._cooled_down()

    def matches_transcript(self, text: str) -> bool:
        text = text.lower().strip()
        words = text.split()
        for wake in settings.wake_words:
            if wake in text:
                return self._cooled_down()
            # Only fuzzy-match short utterances; a long sentence will fuzzy-hit
            # almost any short phrase.
            if len(words) <= 4 and similarity(wake, text) >= settings.wake_word_fuzzy_threshold:
                return self._cooled_down()
        return False

    @staticmethod
    def strip_wake_word(text: str) -> str:
        lowered = text.lower()
        for wake in sorted(settings.wake_words, key=len, reverse=True):
            index = lowered.find(wake)
            if index != -1:
                return text[index + len(wake) :].strip(" ,.!?")
        return text.strip()
