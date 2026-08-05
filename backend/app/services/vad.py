"""Voice activity detection with a Silero -> WebRTC -> energy fallback chain.

The detector is frame-based and stateful: audio arrives from the browser in
30 ms PCM16 frames and `SpeechSegmenter` emits a complete utterance the moment
trailing silence is observed, so nothing ever waits a fixed number of seconds.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np

from ..config import settings

logger = logging.getLogger(__name__)


class FrameDetector(Protocol):
    name: str

    def is_speech(self, frame: np.ndarray) -> bool:
        """`frame` is float32 mono in [-1, 1] at settings.sample_rate."""


class SileroDetector:
    name = "silero"

    def __init__(self) -> None:
        import torch  # type: ignore[import-not-found]
        from silero_vad import load_silero_vad  # type: ignore[import-not-found]

        self._torch = torch
        self._model = load_silero_vad(onnx=True)
        # Silero expects 512-sample windows at 16 kHz; we accumulate frames.
        self._window = 512 if settings.sample_rate == 16000 else 256
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last = False

    def is_speech(self, frame: np.ndarray) -> bool:
        self._buffer = np.concatenate([self._buffer, frame])
        while self._buffer.size >= self._window:
            window, self._buffer = self._buffer[: self._window], self._buffer[self._window :]
            tensor = self._torch.from_numpy(window)
            prob = float(self._model(tensor, settings.sample_rate).item())
            self._last = prob >= settings.vad_speech_threshold
        return self._last


class WebRtcDetector:
    name = "webrtc"

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad  # type: ignore[import-not-found]

        self._vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame: np.ndarray) -> bool:
        pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return self._vad.is_speech(pcm, settings.sample_rate)


class EnergyDetector:
    """Last-resort detector so the pipeline still works with zero extra installs."""

    name = "energy"
    threshold = 0.02

    def is_speech(self, frame: np.ndarray) -> bool:
        return bool(np.sqrt(np.mean(np.square(frame))) > self.threshold)


def build_detector() -> FrameDetector:
    for factory in (SileroDetector, WebRtcDetector):
        try:
            detector = factory()
            logger.info("VAD backend: %s", detector.name)
            return detector
        except Exception as error:  # noqa: BLE001 - optional backends
            logger.warning("VAD backend %s unavailable: %s", factory.__name__, error)
    logger.warning("Falling back to energy VAD")
    return EnergyDetector()


class SpeechSegmenter:
    """Turns a stream of frames into complete utterances."""

    def __init__(self, detector: Optional[FrameDetector] = None) -> None:
        self.detector = detector or build_detector()
        self._frames: list[np.ndarray] = []
        self._prefix: list[np.ndarray] = []
        self._speech_started = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._prefix_frames = max(1, int(300 / settings.vad_frame_ms))

    @property
    def speaking(self) -> bool:
        return self._speech_started

    def reset(self) -> None:
        self._frames.clear()
        self._prefix.clear()
        self._speech_started = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Feed one frame; returns the utterance audio once speech has ended."""
        frame_ms = frame.size / settings.sample_rate * 1000
        is_speech = self.detector.is_speech(frame)

        if not self._speech_started:
            # Keep a rolling pre-roll so the first phoneme is never clipped.
            self._prefix.append(frame)
            if len(self._prefix) > self._prefix_frames:
                self._prefix.pop(0)
            if is_speech:
                self._speech_started = True
                self._frames = [*self._prefix, frame]
                self._speech_ms = frame_ms
            return None

        self._frames.append(frame)
        if is_speech:
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms

        duration = sum(f.size for f in self._frames) / settings.sample_rate
        ended = self._silence_ms >= settings.vad_silence_timeout_ms
        if ended or duration >= settings.vad_max_utterance_seconds:
            audio = np.concatenate(self._frames)
            speech_ms = self._speech_ms
            self.reset()
            if speech_ms < settings.vad_min_speech_ms:
                return None  # cough, click, door slam
            return audio
        return None
