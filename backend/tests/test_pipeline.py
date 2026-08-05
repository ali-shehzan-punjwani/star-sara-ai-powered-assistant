from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.events import AssistantState, ServerEvent
from app.main import app
from app.services import intents
from app.services.pipeline import VoiceSession
from app.services.tts import SentenceChunker
from app.services.vad import EnergyDetector, SpeechSegmenter
from app.services.wakeword import WakeWordDetector


def tone(seconds: float, amplitude: float = 0.4) -> np.ndarray:
    t = np.linspace(0, seconds, int(settings.sample_rate * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def frames(audio: np.ndarray) -> list[np.ndarray]:
    size = int(settings.sample_rate * settings.vad_frame_ms / 1000)
    return [audio[i : i + size] for i in range(0, audio.size - size, size)]


def test_segmenter_emits_utterance_after_trailing_silence():
    segmenter = SpeechSegmenter(EnergyDetector())
    stream = np.concatenate([tone(1.0), np.zeros(settings.sample_rate, dtype=np.float32)])

    emitted = [segmenter.push(frame) for frame in frames(stream)]
    utterances = [audio for audio in emitted if audio is not None]

    assert len(utterances) == 1
    assert utterances[0].size >= settings.sample_rate  # speech plus pre-roll


def test_segmenter_ignores_clicks_shorter_than_min_speech():
    segmenter = SpeechSegmenter(EnergyDetector())
    stream = np.concatenate([tone(0.06), np.zeros(settings.sample_rate, dtype=np.float32)])

    assert all(segmenter.push(frame) is None for frame in frames(stream))


def test_wake_word_matches_fuzzily_and_cools_down():
    detector = WakeWordDetector()

    assert detector.matches_transcript("star sara, what's due today?")
    assert not detector.matches_transcript("star sara again")  # inside cooldown
    assert WakeWordDetector.strip_wake_word("star sara explain aws iam") == "explain aws iam"


def test_sentence_chunker_releases_speakable_chunks():
    chunker = SentenceChunker()
    released: list[str] = []
    for token in "Here is a reasonably long first sentence for you. And a second one. ".split(" "):
        released.extend(chunker.push(token + " "))

    assert released
    assert released[0].endswith(".")
    assert len(released[0]) >= 60


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("add task review the quarterly report", "add_task"),
        ("what tasks do I have", "list_tasks"),
        ("remember that my AWS region is eu-west-1", "remember"),
        ("explain how AWS IAM roles differ from policies", "chat"),
    ],
)
def test_intent_classification(utterance: str, expected: str):
    assert intents.classify(utterance) == expected


async def test_local_intent_turn_streams_reply_and_metrics():
    events: list[ServerEvent] = []

    async def emit(event: ServerEvent) -> None:
        events.append(event)

    session = VoiceSession(emit)
    await session.handle_text("add task ship the STAR SARA launch page", speak=False)
    assert session._turn is not None
    await session._turn

    types = [event.type for event in events]
    assert "token" in types
    assert "reply_done" in types
    assert "metrics" in types
    assert events[-1].state is AssistantState.IDLE


def test_status_endpoint_reports_engines():
    with TestClient(app) as client:
        payload = client.get("/api/status").json()

    assert payload["assistant"] == "STAR SARA"
    assert payload["owner"]["company"] == "STAR Technologies"
    assert "wake_word" in payload["engine"]
