"""The real-time conversation pipeline.

    mic frames -> VAD -> wake word -> faster-whisper -> Groq (streaming)
                                                     -> streaming TTS -> speaker

Every stage hands its output to the next as soon as it has one: the first TTS
sentence is synthesized while Groq is still generating the rest of the reply,
which is what keeps time-to-first-audio around a second.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Optional

import numpy as np

from ..config import settings
from ..core.events import AssistantState, ServerEvent
from . import intents, tts
from .audio import pcm16_to_float32, rms
from .llm import brain
from .memory import store
from .stt import recognizer
from .vad import SpeechSegmenter
from .wakeword import WakeWordDetector

logger = logging.getLogger(__name__)

Emit = Callable[[ServerEvent], Awaitable[None]]


class VoiceSession:
    """One browser connection worth of conversation state."""

    def __init__(self, emit: Emit) -> None:
        self.emit = emit
        self.segmenter = SpeechSegmenter()
        self.wake = WakeWordDetector()
        self.state = AssistantState.IDLE
        self.awake_until = 0.0
        self.always_on = False
        self.mode = settings.accuracy_mode
        self._frame_samples = int(settings.sample_rate * settings.vad_frame_ms / 1000)
        self._pending = np.zeros(0, dtype=np.float32)
        self._turn: Optional[asyncio.Task[None]] = None
        self._wake_detected_at: Optional[float] = None

    # ------------------------------------------------------------------ state

    async def set_state(self, state: AssistantState) -> None:
        if state is not self.state:
            self.state = state
            await self.emit(ServerEvent(type="state", state=state))

    @property
    def in_conversation(self) -> bool:
        return self.always_on or time.monotonic() < self.awake_until

    def configure(self, data: dict) -> None:
        mode = data.get("accuracy_mode")
        if mode in {"fast", "balanced", "accurate"}:
            self.mode = mode
        if "always_on" in data:
            self.always_on = bool(data["always_on"])

    # ------------------------------------------------------------------ audio

    async def feed_audio(self, payload: bytes) -> None:
        """Consume PCM16 mono @ sample_rate from the browser."""
        self._pending = np.concatenate([self._pending, pcm16_to_float32(payload)])

        while self._pending.size >= self._frame_samples:
            frame = self._pending[: self._frame_samples]
            self._pending = self._pending[self._frame_samples :]
            await self._handle_frame(frame)

    async def _handle_frame(self, frame: np.ndarray) -> None:
        if self.state is AssistantState.RESPONDING and rms(frame) > 0.12:
            await self.interrupt()  # barge-in

        if not self.in_conversation and self.wake.acoustic and self.wake.process_frame(frame):
            self._wake_detected_at = time.monotonic()
            self.awake_until = time.monotonic() + settings.followup_window_seconds
            await self.emit(ServerEvent(type="wake", text=settings.assistant_name))
            await self.set_state(AssistantState.LISTENING)

        utterance = self.segmenter.push(frame)
        if self.segmenter.speaking and self.state is AssistantState.IDLE and self.in_conversation:
            await self.set_state(AssistantState.LISTENING)
        if utterance is not None:
            await self._on_utterance(utterance)

    async def _on_utterance(self, audio: np.ndarray) -> None:
        started = time.monotonic()
        transcript = await recognizer.transcribe(audio, self.mode)
        if transcript is None:
            if not self.in_conversation:
                await self.set_state(AssistantState.IDLE)
            return

        text = transcript.text
        if not self.in_conversation:
            # No acoustic wake engine available: gate on the transcript instead.
            if not self.wake.matches_transcript(text):
                return
            self._wake_detected_at = started
            text = self.wake.strip_wake_word(text) or ""
            if not text:
                self.awake_until = time.monotonic() + settings.followup_window_seconds
                await self.emit(ServerEvent(type="wake", text=settings.assistant_name))
                await self.set_state(AssistantState.LISTENING)
                return
        else:
            text = self.wake.strip_wake_word(text) if self.wake.matches_transcript(text) else text

        await self.emit(ServerEvent(type="transcript", text=text))
        await self.handle_text(text, stt_ms=transcript.duration_ms)

    # ------------------------------------------------------------------- turn

    async def handle_text(
        self, text: str, stt_ms: Optional[float] = None, speak: bool = True
    ) -> None:
        await self.interrupt()
        self._turn = asyncio.create_task(self._run_turn(text, stt_ms, speak))

    async def interrupt(self) -> None:
        if self._turn and not self._turn.done():
            self._turn.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn
        self._turn = None

    async def _run_turn(self, text: str, stt_ms: Optional[float], speak: bool) -> None:
        turn_id = uuid.uuid4().hex[:12]
        turn_started = time.monotonic()
        wake_ms = (
            (turn_started - self._wake_detected_at) * 1000 if self._wake_detected_at else None
        )
        self._wake_detected_at = None

        try:
            await self.set_state(AssistantState.THINKING)
            handled = self._handle_intent(text)
            if handled is not None:
                await self.emit(ServerEvent(type="token", text=handled, turn_id=turn_id))
                await self._speak_and_finish(handled, turn_id, turn_started, wake_ms, stt_ms, speak)
                return

            chunker = tts.SentenceChunker()
            speech_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
            first_audio_at: list[float] = []
            speaker = asyncio.create_task(
                self._speech_worker(speech_queue, turn_id, first_audio_at)
            ) if speak else None

            first_token_ms: Optional[float] = None
            reply_parts: list[str] = []
            async for token in brain.stream(text):
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - turn_started) * 1000
                    await self.set_state(AssistantState.RESPONDING)
                reply_parts.append(token)
                await self.emit(ServerEvent(type="token", text=token, turn_id=turn_id))
                if speaker:
                    for chunk in chunker.push(token):
                        await speech_queue.put(chunk)

            if speaker:
                tail = chunker.flush()
                if tail:
                    await speech_queue.put(tail)
                await speech_queue.put(None)
                await speaker

            await self.emit(
                ServerEvent(type="reply_done", text="".join(reply_parts).strip(), turn_id=turn_id)
            )
            await self._finish(
                turn_id,
                turn_started,
                wake_ms,
                stt_ms,
                first_token_ms,
                first_audio_at[0] if first_audio_at else None,
            )
        except asyncio.CancelledError:
            await self.set_state(AssistantState.IDLE)
            raise
        except Exception as error:  # noqa: BLE001 - report, never drop the socket
            logger.exception("Turn failed")
            await self.emit(ServerEvent(type="error", text=str(error), turn_id=turn_id))
            await self.set_state(AssistantState.IDLE)

    async def _speak_and_finish(
        self,
        reply: str,
        turn_id: str,
        turn_started: float,
        wake_ms: Optional[float],
        stt_ms: Optional[float],
        speak: bool,
    ) -> None:
        await self.set_state(AssistantState.RESPONDING)
        first_audio: Optional[float] = None
        if speak:
            async for chunk in tts.synthesize(reply):
                if first_audio is None:
                    first_audio = (time.monotonic() - turn_started) * 1000
                await self.emit(
                    ServerEvent(
                        type="audio", audio=base64.b64encode(chunk).decode(), turn_id=turn_id
                    )
                )
            await self.emit(ServerEvent(type="audio_done", turn_id=turn_id))
        await self.emit(ServerEvent(type="reply_done", text=reply, turn_id=turn_id))
        await self._finish(turn_id, turn_started, wake_ms, stt_ms, 0.0, first_audio)

    async def _speech_worker(
        self, queue: asyncio.Queue[Optional[str]], turn_id: str, first_audio_at: list[float]
    ) -> None:
        started = time.monotonic()
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            async for audio in tts.synthesize(chunk):
                if not first_audio_at:
                    first_audio_at.append((time.monotonic() - started) * 1000)
                await self.emit(
                    ServerEvent(
                        type="audio", audio=base64.b64encode(audio).decode(), turn_id=turn_id
                    )
                )
        await self.emit(ServerEvent(type="audio_done", turn_id=turn_id))

    async def _finish(
        self,
        turn_id: str,
        turn_started: float,
        wake_ms: Optional[float],
        stt_ms: Optional[float],
        first_token_ms: Optional[float],
        first_audio_ms: Optional[float],
    ) -> None:
        self.awake_until = time.monotonic() + settings.followup_window_seconds
        await self.emit(
            ServerEvent(
                type="metrics",
                turn_id=turn_id,
                data={
                    "wake_ms": wake_ms,
                    "stt_ms": stt_ms,
                    "first_token_ms": first_token_ms,
                    "first_audio_ms": first_audio_ms,
                    "total_ms": (time.monotonic() - turn_started) * 1000,
                },
            )
        )
        await self.set_state(AssistantState.IDLE)

    # ---------------------------------------------------------------- intents

    def _handle_intent(self, text: str) -> Optional[str]:
        """Answer locally when possible — zero network latency."""
        intent = intents.classify(text)
        owner = settings.owner_address
        payload = intents.strip_intent_prefix(text, intent)

        if intent == "remember" and payload:
            key, _, value = payload.partition(" is ")
            fact = store.remember(key or payload, value or payload)
            return f"Noted, {owner}. I'll remember that {fact['key']} {fact['value']}".strip()
        if intent == "add_task" and payload:
            store.add_task(payload)
            return f"Added to your list, {owner}: {payload}."
        if intent == "list_tasks":
            pending = store.pending_tasks()
            if not pending:
                return f"Your task list is clear, {owner}."
            listed = "; ".join(task["task"] for task in pending[:5])
            return f"You have {len(pending)} pending tasks, {owner}: {listed}."
        if intent == "tasks_due_today":
            due = store.tasks_due_today()
            if not due:
                return f"Nothing is due today, {owner}."
            return f"Due today, {owner}: " + "; ".join(task["task"] for task in due)
        if intent == "save_note" and payload:
            store.save_note(payload[:48], payload)
            return f"Note saved, {owner}."
        if intent == "list_notes":
            notes = store.notes.get("notes", [])
            if not notes:
                return f"You have no notes yet, {owner}."
            return f"Your latest notes, {owner}: " + "; ".join(n["title"] for n in notes[-5:])
        if intent == "search_notes" and payload:
            found = store.search_notes(payload)
            if not found:
                return f"I couldn't find a note about {payload}, {owner}."
            return "I found: " + "; ".join(n["title"] for n in found[:3])
        if intent == "stop":
            return f"Of course, {owner}."
        return None
