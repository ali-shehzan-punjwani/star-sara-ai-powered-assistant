"""Tests for AudioProcessor (numpy-only DSP helpers)."""

import numpy as np
import pytest


@pytest.fixture
def AP(app_module):
    return app_module.AudioProcessor


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------

def test_normalize_scales_peak_to_target(AP):
    audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    out = AP.normalize(audio, target_peak=0.9)

    assert np.max(np.abs(out)) == pytest.approx(0.9, abs=1e-6)
    assert out.dtype == np.float32


def test_normalize_leaves_near_silence_untouched(AP):
    audio = np.zeros(100, dtype=np.float32)
    out = AP.normalize(audio)
    assert np.array_equal(out, audio)


def test_normalize_handles_empty_array(AP):
    out = AP.normalize(np.array([], dtype=np.float32))
    assert out.size == 0


# --------------------------------------------------------------------------
# trim_silence
# --------------------------------------------------------------------------

def test_trim_silence_removes_leading_and_trailing_quiet(AP):
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)          # 1 second of silence
    audio[8000:8100] = 0.5                           # a loud burst in the middle

    trimmed = AP.trim_silence(audio, sample_rate=sr, pad_ms=10)

    assert trimmed.size < audio.size
    assert np.max(np.abs(trimmed)) == pytest.approx(0.5)


def test_trim_silence_returns_all_silence_unchanged(AP):
    audio = np.full(500, 0.001, dtype=np.float32)    # below threshold everywhere
    out = AP.trim_silence(audio, threshold=0.015)
    assert np.array_equal(out, audio)


def test_trim_silence_handles_empty_array(AP):
    out = AP.trim_silence(np.array([], dtype=np.float32))
    assert out.size == 0


# --------------------------------------------------------------------------
# reduce_noise (noisereduce not installed -> identity fallback)
# --------------------------------------------------------------------------

def test_reduce_noise_is_noop_without_dependency(AP, app_module):
    assert app_module.NOISEREDUCE_AVAILABLE is False
    audio = np.linspace(-1, 1, 8000, dtype=np.float32)
    out = AP.reduce_noise(audio, sample_rate=16000)
    assert np.array_equal(out, audio)


def test_reduce_noise_short_clip_is_noop(AP):
    audio = np.ones(10, dtype=np.float32)
    out = AP.reduce_noise(audio, sample_rate=16000)
    assert np.array_equal(out, audio)


# --------------------------------------------------------------------------
# clean (full pipeline)
# --------------------------------------------------------------------------

def test_clean_pipeline_trims_and_normalizes(AP):
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    audio[8000:8300] = 0.25

    out = AP.clean(audio, sample_rate=sr)

    assert out.dtype == np.float32
    assert out.size < audio.size                     # trimmed
    assert np.max(np.abs(out)) == pytest.approx(0.9, abs=1e-6)  # normalized
