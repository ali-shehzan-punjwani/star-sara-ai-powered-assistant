"""Cheap DSP applied before transcription: trim, normalize, high-pass."""

from __future__ import annotations

import numpy as np

from ..config import settings

_SILENCE_THRESHOLD = 0.012
_PAD_MS = 120


def pcm16_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak < 1e-6:
        return audio
    return audio * (target_peak / peak)


def trim_silence(audio: np.ndarray) -> np.ndarray:
    loud = np.where(np.abs(audio) > _SILENCE_THRESHOLD)[0]
    if loud.size == 0:
        return audio
    pad = int(settings.sample_rate * _PAD_MS / 1000)
    return audio[max(0, loud[0] - pad) : min(audio.size, loud[-1] + pad)]


def high_pass(audio: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """One-pole pre-emphasis; removes rumble without pulling in scipy."""
    if audio.size < 2:
        return audio
    filtered = np.empty_like(audio)
    filtered[0] = audio[0]
    filtered[1:] = audio[1:] - alpha * audio[:-1]
    return filtered


def clean(audio: np.ndarray) -> np.ndarray:
    return normalize(trim_silence(high_pass(audio.astype(np.float32))))


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))
