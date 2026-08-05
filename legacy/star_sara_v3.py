"""
==============================================================================
STAR SARA v3.0
Smart AI Response Assistant — Futuristic GUI Edition (Enhanced Intelligence)
==============================================================================

This file is an ENHANCEMENT of your existing STAR SARA v2.0. The GUI classes
(StarCoreWidget, InfoPanel, FuturisticGUI), overall architecture, wake words,
file layout, and public behavior are UNCHANGED. What changed and why is
explained in the accompanying chat message, section by section.

New dependencies you need to install:
    pip install rapidfuzz webrtcvad numpy scipy noisereduce

If any of these are missing, the code below degrades gracefully (falls back
to the old fixed-duration recording / exact wake-word matching) rather than
crashing, so you can adopt pieces incrementally.
==============================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

import os
ffmpeg_path = os.path.join(os.path.dirname(__file__), "ffmpeg", "bin")
os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
import sys
import json
import math
import time
import random
import asyncio
import tempfile
import traceback
import re
import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

from dotenv import load_dotenv
from groq import Groq

import numpy as np
import whisper
import sounddevice as sd
import soundfile as sf
import edge_tts
import pygame

from star_sara_utils import (
    RAPIDFUZZ_AVAILABLE,
    format_numbered_list,
    load_json,
    load_json_collection,
    log_error,
    log_warning,
    now_iso,
    polar_point,
    save_json,
    strip_phrases,
    text_similarity,
)

# ---- optional / graceful-degradation dependencies ----

try:
    import webrtcvad
    WEBRTCVAD_AVAILABLE = True
except Exception:
    WEBRTCVAD_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except Exception:
    NOISEREDUCE_AVAILABLE = False

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, Slot, QObject, QRectF, QPointF
)
from PySide6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QRadialGradient,
    QLinearGradient, QPainterPath, QPolygonF
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QFrame, QSizePolicy, QTextEdit
)


# ==============================================================================
# ENVIRONMENT
# ==============================================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_CLIENT: Optional[Groq] = None
GROQ_AVAILABLE = False

if GROQ_API_KEY:
    try:
        GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
        GROQ_AVAILABLE = True
    except Exception as error:
        log_error("Could not initialize Groq client", error)
else:
    print("[WARNING] GROQ_API_KEY missing in .env — AI brain will run in offline mode.")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

ASSISTANT_NAME = "STAR SARA"
OWNER_ADDRESS = "Shehzan Sir"
GROQ_MODEL = "llama-3.3-70b-versatile"

WAKE_WORDS = ["star sara", "sara" , "starzera" , "hello" , "star"]

TTS_VOICE = "en-US-AriaNeural"
WHISPER_MODEL_NAME = "medium"

LISTEN_DURATION_WAKE = 4        # seconds, fixed-duration fallback (no VAD available)
LISTEN_DURATION_COMMAND = 6     # seconds, fixed-duration fallback (no VAD available)

# --- new tunables (speech recognition / VAD) ---
SAMPLE_RATE = 16000
VAD_FRAME_MS = 30                    # webrtcvad requires 10/20/30 ms frames
VAD_AGGRESSIVENESS = 2               # 0 (permissive) .. 3 (aggressive)
VAD_SILENCE_TIMEOUT_MS = 700         # stop recording after this much trailing silence
VAD_MAX_RECORD_SECONDS = 12          # hard safety cap
VAD_MIN_SPEECH_MS = 250              # ignore clips shorter than this (coughs, clicks)
WHISPER_NO_SPEECH_THRESHOLD = 0.6    # discard transcriptions Whisper itself thinks are silence

# --- new tunables (wake word) ---
WAKE_WORD_FUZZY_THRESHOLD = 90        # rapidfuzz score (0-100) — raised from 82: 82 let
                                       # things like "starzada" fuzzy-match "star sara"
WAKE_WORD_MAX_WORDS_FOR_FUZZY = 4     # only fuzzy-check short utterances, not whole sentences
WAKE_WORD_COOLDOWN_SECONDS = 2.5      # ignore repeat wake-word triggers within this window

# --- new tunables (conversation flow) ---
CONVERSATION_FOLLOWUP_TIMEOUT = 6    # after answering, listen this long for a wake-word-free follow-up
CONVERSATION_HISTORY_TURNS = 6       # how many past user/assistant turn PAIRS to keep in context

# --- new tunables (memory) ---
MEMORY_MAX_FACTS = 300
MEMORY_DEDUPE_FUZZY_THRESHOLD = 88
MEMORY_TOP_K_FOR_CONTEXT = 8
MEMORY_DECAY_DAYS = 30
MEMORY_DECAY_MIN_ACCESS = 1

# --- barge-in ---
# WHY default False: without acoustic echo cancellation, sampling the mic
# while TTS plays through the SAME device (e.g. a Bluetooth headset) will
# likely pick up STAR SARA's own voice and interrupt her mid-sentence. Only
# turn this on once you've confirmed your mic/speaker setup doesn't leak
# playback back into the mic (e.g. wired headphones with a separate mic, or
# you've tested it and it behaves).
ENABLE_BARGE_IN = False
BARGE_IN_ENERGY_THRESHOLD = 0.035    # RMS amplitude above which we treat mic as "user talking"
BARGE_IN_POLL_MS = 100


# ==============================================================================
# FILE SYSTEM
# ==============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

USER_FILE = os.path.join(BASE_DIR, "user_data.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
TASKS_FILE = os.path.join(BASE_DIR, "tasks.json")
NOTES_FILE = os.path.join(BASE_DIR, "notes.json")


# ==============================================================================
# OWNER-FACING PHRASES
# ==============================================================================

def _apology(detail: str) -> str:
    return f"Sorry {OWNER_ADDRESS}, {detail}"


NO_NOTES_RESPONSE = f"You do not have any saved notes, {OWNER_ADDRESS}."


# ==============================================================================
# AUDIO PROCESSOR — normalization, silence trimming, light noise reduction
# ==============================================================================

class AudioProcessor:
    """
    WHY: Whisper's accuracy on a laptop mic drops a lot when clips are too
    quiet, have DC offset, have dead air at the start/end, or carry steady
    fan/hum noise. All three fixes here are cheap (numpy-only, run in a few
    ms) and consistently raise transcription accuracy before the audio ever
    reaches Whisper.
    """

    @staticmethod
    def normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
        audio = audio.astype(np.float32)
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak < 1e-6:
            return audio
        return audio * (target_peak / peak)

    @staticmethod
    def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                      threshold: float = 0.015, pad_ms: int = 150) -> np.ndarray:
        if audio.size == 0:
            return audio
        amplitude = np.abs(audio.flatten())
        above = np.where(amplitude > threshold)[0]
        if above.size == 0:
            return audio  # all silence — let the caller decide what to do with it
        pad = int(sample_rate * pad_ms / 1000)
        start = max(0, above[0] - pad)
        end = min(len(audio), above[-1] + pad)
        return audio[start:end]

    @staticmethod
    def reduce_noise(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """
        Estimates a noise profile from the first 0.3s of the clip (usually
        room hum before speech starts) and subtracts it. Falls back to a
        no-op if `noisereduce` isn't installed rather than failing.
        """
        if not NOISEREDUCE_AVAILABLE or audio.size < sample_rate * 0.3:
            return audio
        try:
            flat = audio.flatten()
            noise_clip = flat[: int(sample_rate * 0.3)]
            return nr.reduce_noise(y=flat, sr=sample_rate, y_noise=noise_clip, stationary=True)
        except Exception as error:
            log_warning("noisereduce failed, using raw audio", error)
            return audio

    @classmethod
    def clean(cls, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
        """Full pipeline: noise reduce -> trim -> normalize."""
        audio = cls.reduce_noise(audio, sample_rate)
        trimmed = cls.trim_silence(audio, sample_rate)
        if trimmed.size > 0:
            audio = trimmed
        return cls.normalize(audio)


# ==============================================================================
# AI ENGINE — Groq brain + short/long-term memory, tasks, notes
# ==============================================================================

class AIEngine:
    """
    Owns all persisted state (user profile, memory, tasks, notes) and talks
    to the Groq Llama model.

    CHANGES vs v2:
    - Memory facts now carry importance / timestamp / access metadata instead
      of being a flat {key, value} list, so we can rank, dedupe, and decay them.
    - `build_context()` only injects the TOP-K most relevant memories for the
      current message instead of the entire memory file — this keeps context
      focused (less hallucination / drift) and keeps prompts small as memory
      grows, instead of degrading over time.
    - A rolling `conversation_history` deque gives real multi-turn context
      (previously every command was answered with zero memory of what was
      just said).
    """

    def __init__(self):
        self.user_data = load_json(USER_FILE, {
            "assistant": {
                "name": ASSISTANT_NAME,
                "owner_name": OWNER_ADDRESS,
                "greeting": f"Yes, {OWNER_ADDRESS}. How may I assist you today?",
            },
            "identity": {
                "full_name": "Ali Shehzan Punjwani",
                "preferred_name": "Shehzan",
                "country": "Pakistan",
                "city": "Karachi",
            },
            "career": {
                "current_focus": "Cloud Security",
                "target_role": "Chief Information Security Officer (CISO)",
            },
        })

        # WHY a separate "safe" copy: user_data.json can carry identity/family/
        # contact fields that shouldn't be sent to a third-party API (Groq) on
        # every single message just to answer things like "what's the weather".
        # Sections in _EXCLUDED_FROM_LLM stay local-only; everything else
        # still goes to the model so it can personalize answers.
        self._PROFILE_SECTIONS_EXCLUDED_FROM_LLM = {"identity", "contact", "family"}

        self.memory = load_json(MEMORY_FILE, {"facts": []})
        self._migrate_memory_schema()
        self._decay_memory()

        # load_json_collection returns whatever is actually on disk while
        # guaranteeing the collection key exists — tasks.json/notes.json
        # written by an earlier version (or hand-edited, or emptied to {})
        # would otherwise raise KeyError on the first write.
        self.tasks = load_json_collection(TASKS_FILE, "tasks")
        self.notes = load_json_collection(NOTES_FILE, "notes")

        # short-term multi-turn context (not persisted — resets per session,
        # anything worth keeping long-term should go through `remember()`)
        self.conversation_history: deque = deque(maxlen=CONVERSATION_HISTORY_TURNS * 2)

    def _llm_safe_profile(self) -> dict:
        """
        Returns a copy of user_data with sections/keys that shouldn't be sent
        to a third-party API stripped out — contact info and family are
        dropped entirely (rarely relevant to answering a question and
        needlessly exposes other people's names); date_of_birth/age/religion/
        gender are stripped from identity specifically. Everything else
        (career, skills, projects, goals, preferences, safe identity fields
        like preferred_name/city/country/languages/motto) still goes through
        so responses stay personalized.
        """
        sensitive_identity_keys = {"date_of_birth", "age", "religion", "gender"}
        safe: Dict = {}
        for section, content in self.user_data.items():
            if section in self._PROFILE_SECTIONS_EXCLUDED_FROM_LLM:
                if section == "identity" and isinstance(content, dict):
                    # keep the non-sensitive identity fields (name, city, languages, motto...)
                    safe[section] = {k: v for k, v in content.items() if k not in sensitive_identity_keys}
                continue
            safe[section] = content
        return safe

    # ---------------- memory schema migration ----------------

    def _migrate_memory_schema(self) -> None:
        """Upgrades old {key, value} facts to the new metadata schema in place."""
        changed = False
        for fact in self.memory.get("facts", []):
            if "importance" not in fact:
                fact["importance"] = 3  # 1 (trivial) .. 5 (critical), default mid
                changed = True
            if "created_at" not in fact:
                fact["created_at"] = now_iso()
                changed = True
            if "last_accessed" not in fact:
                fact["last_accessed"] = fact["created_at"]
                changed = True
            if "access_count" not in fact:
                fact["access_count"] = 0
                changed = True
        if changed:
            save_json(MEMORY_FILE, self.memory)

    @staticmethod
    def _touch_fact(fact: Dict) -> None:
        """Records an access on a fact — feeds ranking and decay."""
        fact["access_count"] = fact.get("access_count", 0) + 1
        fact["last_accessed"] = now_iso()

    # ---------------- memory: write ----------------

    def remember(self, key: str, value: str, importance: int = 3) -> str:
        """
        Stores a fact, but first checks for a near-duplicate existing fact
        (fuzzy match on the value text) and updates it instead of appending —
        this is what stops memory.json from filling up with the same fact
        phrased three slightly different ways.
        """
        facts = self.memory.setdefault("facts", [])

        for fact in facts:
            if text_similarity(fact.get("value", ""), value) >= MEMORY_DEDUPE_FUZZY_THRESHOLD:
                fact["value"] = value
                fact["importance"] = max(fact.get("importance", 3), importance)
                fact["last_accessed"] = now_iso()
                save_json(MEMORY_FILE, self.memory)
                return f"I updated what I remembered, {OWNER_ADDRESS}."

        created_at = now_iso()
        facts.append({
            "key": key,
            "value": value,
            "importance": importance,
            "created_at": created_at,
            "last_accessed": created_at,
            "access_count": 0,
        })

        # keep the file bounded — drop the least important, least recently
        # accessed fact rather than growing forever
        if len(facts) > MEMORY_MAX_FACTS:
            facts.sort(key=lambda f: (f["importance"], f["last_accessed"]))
            facts.pop(0)

        save_json(MEMORY_FILE, self.memory)
        return f"I will remember that, {OWNER_ADDRESS}."

    # ---------------- memory: read / rank ----------------

    def recall(self, key: str) -> Optional[str]:
        for item in self.memory.get("facts", []):
            if item.get("key") == key:
                self._touch_fact(item)
                return item.get("value")
        return None

    def relevant_memories(self, query: str, top_k: int = MEMORY_TOP_K_FOR_CONTEXT) -> List[Dict]:
        """
        Ranks stored facts against the current query by a blend of text
        relevance and importance, and returns only the top-K. This is the
        piece that lets memory scale — instead of dumping every fact into
        every prompt, only what's actually relevant to *this* question goes
        in, and touched facts get their access stats bumped (used by decay).
        """
        facts = self.memory.get("facts", [])
        if not facts:
            return []

        scored: List[Tuple[float, Dict]] = []
        for fact in facts:
            relevance = text_similarity(query, fact.get("value", ""))
            key_relevance = text_similarity(query, fact.get("key", ""))
            score = max(relevance, key_relevance) + fact.get("importance", 3) * 4
            scored.append((score, fact))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [fact for score, fact in scored[:top_k] if score > 20]

        for fact in top:
            self._touch_fact(fact)

        if top:
            save_json(MEMORY_FILE, self.memory)

        return top

    def _decay_memory(self) -> None:
        """
        Runs once at startup: facts that are old, low-importance, and rarely
        accessed get quietly dropped. WHY: without this, memory.json becomes
        an ever-growing junk drawer and low-value facts start crowding out
        useful ones in `relevant_memories()`.
        """
        facts = self.memory.get("facts", [])
        if not facts:
            return

        cutoff = datetime.now() - timedelta(days=MEMORY_DECAY_DAYS)
        kept = []
        removed = 0
        for fact in facts:
            try:
                last_accessed = datetime.fromisoformat(fact.get("last_accessed", fact.get("created_at")))
            except Exception:
                last_accessed = datetime.now()

            is_stale = (
                last_accessed < cutoff
                and fact.get("importance", 3) <= 2
                and fact.get("access_count", 0) <= MEMORY_DECAY_MIN_ACCESS
            )
            if is_stale:
                removed += 1
            else:
                kept.append(fact)

        if removed:
            self.memory["facts"] = kept
            save_json(MEMORY_FILE, self.memory)
            print(f"[INFO] Memory decay removed {removed} stale low-importance fact(s).")

    # ---------------- conversation history ----------------

    def add_turn(self, role: str, content: str) -> None:
        self.conversation_history.append({"role": role, "content": content})

    # ---------------- tasks ----------------

    def add_task(self, task_text: str, priority: str = "normal", due: Optional[str] = None) -> None:
        self.tasks["tasks"].append({
            "task": task_text,
            "priority": priority,
            "status": "pending",
            "due": due,  # ISO date string or None — powers "calendar awareness" below
            "created_at": now_iso(),
        })
        save_json(TASKS_FILE, self.tasks)

    def get_pending_tasks(self) -> List[Dict]:
        return [t for t in self.tasks.get("tasks", []) if t.get("status") == "pending"]

    def get_tasks_due_today(self) -> List[Dict]:
        today = datetime.now().date().isoformat()
        return [t for t in self.get_pending_tasks() if t.get("due") == today]

    def complete_task(self, index: int) -> bool:
        try:
            self.tasks["tasks"][index]["status"] = "completed"
            save_json(TASKS_FILE, self.tasks)
            return True
        except Exception:
            return False

    def format_tasks(self) -> str:
        pending = self.get_pending_tasks()
        if not pending:
            return f"You have no pending tasks, {OWNER_ADDRESS}."

        def render(task: Dict) -> str:
            due_note = f", due {task['due']}" if task.get("due") else ""
            return f"{task['task']}{due_note}"

        return format_numbered_list(
            pending,
            header=f"You have {len(pending)} pending tasks. ",
            label="Task",
            render=render,
        )

    def format_due_today(self) -> str:
        due = self.get_tasks_due_today()
        if not due:
            return f"Nothing is due today, {OWNER_ADDRESS}."

        return format_numbered_list(
            due,
            header=f"You have {len(due)} task(s) due today. ",
            label="Task",
            render=lambda task: task["task"],
        )

    # ---------------- notes ----------------

    def save_note(self, title: str, content: str) -> None:
        self.notes["notes"].append({
            "title": title,
            "content": content,
            "created_at": now_iso(),
        })
        save_json(NOTES_FILE, self.notes)

    def format_notes(self) -> str:
        notes = self.notes.get("notes", [])
        if not notes:
            return NO_NOTES_RESPONSE

        return format_numbered_list(
            notes,
            header=f"You have {len(notes)} saved notes. ",
            label="Note",
            render=lambda note: f"{note['title']}. {note['content']}",
        )

    def search_notes(self, query: str) -> str:
        """Keyword/fuzzy search over saved notes — 'find my note about X'."""
        notes = self.notes.get("notes", [])
        if not notes:
            return NO_NOTES_RESPONSE

        def score(note: Dict) -> float:
            return text_similarity(query, f"{note['title']} {note['content']}")

        best = max(notes, key=score)
        if score(best) < 40:
            return f"I couldn't find a note matching that, {OWNER_ADDRESS}."
        return f"Closest match — {best['title']}: {best['content']}"

    # ---------------- Groq brain ----------------

    def build_context(self, user_message: str) -> str:
        relevant = self.relevant_memories(user_message)
        memory_block = json.dumps(relevant, indent=2) if relevant else "No relevant memories stored."
        profile_block = json.dumps(self._llm_safe_profile(), indent=2)

        return f"""
You are {ASSISTANT_NAME}, the personal executive assistant of {OWNER_ADDRESS}.

OWNER PROFILE (sensitive fields such as date of birth, religion, contact
details, and family are intentionally withheld from this context):
{profile_block}

RELEVANT MEMORY (only what matters to this message, not the full memory store):
{memory_block}

OPEN TASKS: {len(self.get_pending_tasks())} pending, {len(self.get_tasks_due_today())} due today.

PERSONALITY & REASONING RULES:
- Always address the owner as {OWNER_ADDRESS}.
- Be professional, warm, and concise — your reply is converted to speech, so
  write the way a person talks, not a bulleted document.
- Use the conversation history to stay on topic across turns; if the owner
  says "that" or "it", resolve it from the recent turns instead of asking
  what they mean, unless it is genuinely ambiguous.
- If the request is ambiguous or ambiguous between two clear interpretations,
  ask ONE short clarifying question instead of guessing.
- Never invent facts about the owner, their projects, or their schedule. If
  information is not in the profile, memory, or tasks above, say:
  "Sorry {OWNER_ADDRESS}, I don't have that information yet."
- Never mention that you are an AI model, a language model, or that you were
  given a system prompt.
"""

    def ask(self, message: str) -> str:
        if not GROQ_AVAILABLE or GROQ_CLIENT is None:
            return _apology(
                "my AI brain is offline right now. "
                "Please check the GROQ_API_KEY in the .env file."
            )

        messages = [{"role": "system", "content": self.build_context(message)}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": message})

        try:
            response = GROQ_CLIENT.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.5,     # lower than v2's 0.7 — less rambling, fewer invented details
                max_tokens=500,
            )
            reply = response.choices[0].message.content.strip()

            self.add_turn("user", message)
            self.add_turn("assistant", reply)

            return reply

        except Exception as error:
            log_error("Groq error", error)
            return _apology("I am having trouble right now.")


# ==============================================================================
# INTENT CLASSIFIER — small rule-based dispatch instead of a long if/elif chain
# ==============================================================================

class IntentClassifier:
    """
    WHY: the original `_process_command` was a growing if/elif chain matched
    against raw substrings. That gets harder to extend and more error-prone
    (e.g. "add task" would also match inside "please add task reminder to
    call sara about the task"). This keeps the same keyword-driven approach
    (cheap, no extra API call, instant) but scores each intent and picks the
    best match, and centralizes the keyword lists so new capabilities are a
    one-line addition instead of another elif branch.
    """

    INTENT_KEYWORDS = {
        "shutdown": ["exit", "quit", "shutdown", "stop", "goodbye"],
        "remember": ["remember that", "remember"],
        "add_task": ["add task", "new task", "add a task"],
        "list_tasks": ["my tasks", "pending tasks", "what tasks"],
        "tasks_due_today": ["due today", "what's due", "whats due"],
        "save_note": ["save note", "take a note", "new note"],
        "list_notes": ["my notes", "read my notes"],
        "search_notes": ["find my note", "search notes", "note about"],
    }

    FUZZY_SCORE_THRESHOLD = 85
    FUZZY_MAX_LENGTH_RATIO = 1.6   # command can be at most this many times longer than the phrase

    @classmethod
    def trigger_phrases(cls, intent: str) -> List[str]:
        """
        The phrases that route to an intent — also what handlers strip out of
        the command to get the payload ("add task call the bank" -> "call the
        bank"), so the keyword lists live in exactly one place.
        """
        return cls.INTENT_KEYWORDS.get(intent, [])

    @classmethod
    def classify(cls, command: str) -> str:
        command = command.lower().strip()
        command_words = command.split()
        best_intent, best_score = "chat", 0.0

        for intent, phrases in cls.INTENT_KEYWORDS.items():
            for phrase in phrases:
                # WHY word-boundary instead of plain substring: "stop" in
                # command also matches inside "stopping" or "nonstop" — same
                # bug class as the sara/sarah wake-word issue, except here it
                # would shut the whole assistant down mid-sentence.
                if re.search(r"\b" + re.escape(phrase) + r"\b", command):
                    return intent

                # WHY the length-ratio gate: without it, a short phrase like
                # "search notes" fuzzy-matches a long unrelated sentence like
                # "search for the first one" well enough to misfire — this
                # was silently routing free-form chat into the wrong intent.
                phrase_words = phrase.split()
                if len(command_words) > len(phrase_words) * cls.FUZZY_MAX_LENGTH_RATIO:
                    continue

                score = text_similarity(phrase, command)
                if score > best_score:
                    best_intent, best_score = intent, score

        return best_intent if best_score >= cls.FUZZY_SCORE_THRESHOLD else "chat"


# ==============================================================================
# VOICE ENGINE — Whisper STT + VAD + fuzzy wake word + Edge-TTS + barge-in
# ==============================================================================

class VoiceEngine:
    """
    Handles microphone capture + transcription, wake word matching, and
    text-to-speech playback. Designed to run entirely inside a background
    QThread so it never blocks the GUI.

    CHANGES vs v2:
    - `listen()` now records with Voice Activity Detection when webrtcvad is
      installed: it stops as soon as the owner stops talking instead of
      always waiting out a fixed 4-6 second window. That alone is the single
      biggest latency win in this file (no more waiting through dead air).
    - Captured audio is normalized / trimmed / noise-reduced before Whisper
      ever sees it (see AudioProcessor).
    - Whisper decode params are tuned to cut hallucinated text on silence and
      forced to English.
    - `contains_wake_word` is fuzzy (rapidfuzz) so "star sarah", "star sera",
      slightly slurred audio, etc. still trigger — with a cooldown so a
      single utterance can't double-trigger.
    - `speak()` now supports barge-in: while STAR SARA is talking, it also
      samples the mic; if it detects the owner start talking over her, she
      stops immediately instead of finishing the sentence.
    """

    def __init__(self):
        self._model = None
        self._vad = webrtcvad.Vad(VAD_AGGRESSIVENESS) if WEBRTCVAD_AVAILABLE else None
        self._last_wake_trigger_time = 0.0
        pygame.mixer.init()

    def load_model(self) -> None:
        print("[INFO] Loading Whisper model...")
        self._model = whisper.load_model(WHISPER_MODEL_NAME)
        print("[INFO] Whisper model loaded successfully.")

    # ---------------- speech to text ----------------

    def listen(self, duration: int = LISTEN_DURATION_WAKE, allow_vad: bool = True) -> Optional[str]:
        """
        Records audio (VAD-based if available, else fixed-duration fallback),
        cleans it, and transcribes it with Whisper. `duration` is the max
        time to wait for the owner to START talking (not total recording
        length) — once speech starts, VAD keeps recording until it detects
        the trailing silence, regardless of `duration`.
        """
        if allow_vad and WEBRTCVAD_AVAILABLE:
            audio = self._record_with_vad(max_wait_seconds=duration)
        else:
            audio = self._record_fixed_duration(duration)

        if audio is None or audio.size == 0:
            return None

        audio = AudioProcessor.clean(audio, SAMPLE_RATE)
        if audio.size < SAMPLE_RATE * (VAD_MIN_SPEECH_MS / 1000):
            return None  # too short to be real speech — likely a click/cough

        return self._transcribe(audio)

    def _record_fixed_duration(self, duration: int) -> Optional[np.ndarray]:
        try:
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            return audio.flatten()
        except Exception:
            traceback.print_exc()
            return None

    def _record_with_vad(self, max_wait_seconds: float = VAD_MAX_RECORD_SECONDS) -> Optional[np.ndarray]:
        """
        Streams the mic in small frames and stops once it has seen speech
        followed by VAD_SILENCE_TIMEOUT_MS of silence — this is what lets
        STAR SARA respond as soon as the owner finishes a sentence instead of
        waiting out a fixed window.

        `max_wait_seconds` bounds how long we'll wait for speech to START
        (this is what makes CONVERSATION_FOLLOWUP_TIMEOUT actually mean
        something — previously this method ignored the caller's timeout
        entirely and always waited up to VAD_MAX_RECORD_SECONDS). Once
        speech has started, recording still runs up to VAD_MAX_RECORD_SECONDS
        total regardless of max_wait_seconds, so a long sentence isn't cut off.
        """
        frame_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
        silence_frames_needed = int(VAD_SILENCE_TIMEOUT_MS / VAD_FRAME_MS)
        max_wait_frames = int(max_wait_seconds * 1000 / VAD_FRAME_MS)
        max_total_frames = int(VAD_MAX_RECORD_SECONDS * 1000 / VAD_FRAME_MS)

        collected: List[np.ndarray] = []
        speech_started = False
        silence_run = 0
        frames_seen = 0

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                                 blocksize=frame_samples) as stream:
                while frames_seen < max_total_frames:
                    frame, _ = stream.read(frame_samples)
                    frames_seen += 1
                    pcm_bytes = frame.tobytes()

                    is_speech = self._vad.is_speech(pcm_bytes, SAMPLE_RATE)
                    collected.append(frame.flatten())

                    if is_speech:
                        speech_started = True
                        silence_run = 0
                    elif speech_started:
                        silence_run += 1
                        if silence_run >= silence_frames_needed:
                            break
                    elif frames_seen >= max_wait_frames:
                        # nobody started talking within the allotted wait —
                        # give up instead of listening the full 12s every time
                        break
        except Exception:
            traceback.print_exc()
            return None

        if not speech_started or not collected:
            return None

        audio = np.concatenate(collected).astype(np.float32) / 32768.0
        return audio

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        try:
            result = self._model.transcribe(
                audio,
                language="en",              # force English — avoids Whisper guessing the wrong language on short clips
                fp16=False,
                temperature=0.0,            # deterministic decoding, fewer wild guesses
                condition_on_previous_text=False,  # stops one bad guess from poisoning the next
                no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            )

            # Whisper sometimes "transcribes" background noise into fake
            # phrases; segment-level no_speech_prob catches most of that.
            segments = result.get("segments", [])
            if segments and all(seg.get("no_speech_prob", 0) > WHISPER_NO_SPEECH_THRESHOLD for seg in segments):
                return None

            text = result["text"].strip().lower()
            if not text or self._is_likely_hallucination(text):
                return None
            return text

        except Exception:
            traceback.print_exc()
            return None

    @staticmethod
    def _is_likely_hallucination(text: str) -> bool:
        """
        WHY: on quiet/noisy audio Whisper often "confidently" outputs a
        stutter loop like "hello, hello, hello." or "hello? hello?" instead
        of admitting no_speech — that's not caught by no_speech_prob because
        Whisper genuinely thinks it heard something. If the clip collapses
        to one or two unique words repeated several times, treat it as noise.
        """
        tokens = re.findall(r"[a-z']+", text)
        if len(tokens) < 3:
            return False
        unique = set(tokens)
        return len(unique) <= 2

    # ---------------- wake word ----------------

    def contains_wake_word(self, text: Optional[str]) -> bool:
        """
        Word-boundary exact match first, fuzzy match as a fallback — with a
        cooldown. WHY word-boundary instead of plain substring: `"sara" in
        text` also matches inside "sarah", "sarasota", etc. (sara IS the
        first four letters of sarah) — that was firing false activations.
        `\\bsara\\b` requires sara to be its own word. WHY fuzzy at all:
        Whisper on a laptop mic will still sometimes render "star sara" as
        "star sera" / "starsara" with no word boundary Whisper would produce
        anyway — so fuzzy is a narrower fallback, restricted to short
        utterances so it can't fire in the middle of an unrelated sentence.
        WHY cooldown: a single continuous utterance can otherwise be picked
        up twice by overlapping listen() windows and double-trigger.
        """
        if not text:
            return False

        now = time.time()
        if now - self._last_wake_trigger_time < WAKE_WORD_COOLDOWN_SECONDS:
            return False

        lowered = text.lower().strip()
        word_count = len(lowered.split())

        for word in WAKE_WORDS:
            pattern = r"\b" + re.escape(word) + r"\b"
            if re.search(pattern, lowered):
                self._last_wake_trigger_time = now
                return True

        if RAPIDFUZZ_AVAILABLE and word_count <= WAKE_WORD_MAX_WORDS_FOR_FUZZY:
            if text_similarity("star sara", lowered) >= WAKE_WORD_FUZZY_THRESHOLD:
                self._last_wake_trigger_time = now
                return True

        return False

    @staticmethod
    def remove_wake_word(text: str) -> str:
        command = text.lower()
        for word in WAKE_WORDS:
            command = command.replace(word, "")
        return command.strip()

    # ---------------- text to speech ----------------

    async def _generate_voice(self, text: str) -> Optional[str]:
        try:
            communicate = edge_tts.Communicate(text=text, voice=TTS_VOICE)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file:
                audio_file = file.name
            await communicate.save(audio_file)
            return audio_file
        except Exception as error:
            log_error("Edge-TTS error", error)
            return None

    def speak(self, text: str, allow_barge_in: bool = ENABLE_BARGE_IN) -> bool:
        """
        Speaks the given text. Blocking — call from the worker thread only.
        Returns True if it played to completion, False if it was interrupted
        (barge-in) — the caller can use that to immediately start listening
        again instead of waiting for the sentence to finish.
        """
        if not text:
            return True

        print(f"{ASSISTANT_NAME}: {text}")

        interrupted = threading.Event()
        stop_monitor = threading.Event()

        def _monitor_for_barge_in():
            """Samples short mic bursts while TTS plays; stops playback on speech."""
            try:
                block = int(SAMPLE_RATE * BARGE_IN_POLL_MS / 1000)
                while not stop_monitor.is_set():
                    chunk = sd.rec(block, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
                    sd.wait()
                    rms = float(np.sqrt(np.mean(np.square(chunk)))) if chunk.size else 0.0
                    if rms > BARGE_IN_ENERGY_THRESHOLD:
                        interrupted.set()
                        pygame.mixer.music.stop()
                        return
            except Exception:
                pass  # barge-in is a nice-to-have; never let it crash playback

        try:
            audio_file = asyncio.run(self._generate_voice(text))

            if audio_file:
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()

                monitor_thread = None
                if allow_barge_in:
                    monitor_thread = threading.Thread(target=_monitor_for_barge_in, daemon=True)
                    monitor_thread.start()

                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)

                stop_monitor.set()
                if monitor_thread:
                    monitor_thread.join(timeout=0.5)

                pygame.mixer.music.unload()
                os.remove(audio_file)

            return not interrupted.is_set()

        except Exception as error:
            log_error("Voice playback failed", error)
            return True


# ==============================================================================
# ASSISTANT WORKER — background thread running the full voice loop
# ==============================================================================

class AssistantWorker(QThread):
    """
    Runs the STAR SARA listen -> think -> speak loop on a background thread
    so the GUI animations never freeze.

    CHANGES vs v2:
    - Uses IntentClassifier for dispatch instead of a long if/elif chain.
    - After answering, listens briefly WITHOUT requiring the wake word again
      ("continuous conversation mode") — if the owner keeps talking, the
      conversation flows naturally; if not, it falls back to wake-word idle.
    - Reminders/tasks-due-today wired in as new capabilities.
    """

    state_changed = Signal(str)       # "loading" | "idle" | "listening" | "processing" | "speaking"
    owner_said = Signal(str)          # transcribed command text
    assistant_said = Signal(str)      # spoken response text
    error_occurred = Signal(str)
    shutdown_requested = Signal()

    def __init__(self, ai_engine: AIEngine, voice_engine: VoiceEngine, parent=None):
        super().__init__(parent)
        self.ai = ai_engine
        self.voice = voice_engine
        self._running = True

    def stop(self) -> None:
        self._running = False
        # WHY sd.stop(): sd.rec() / InputStream.read() block until audio is
        # captured — just flipping _running doesn't interrupt them, which is
        # why closing the window used to let one more full turn (heard +
        # spoken) run before the app actually exited. This aborts the
        # in-flight capture immediately so the loop can see _running=False
        # on its next check.
        try:
            sd.stop()
        except Exception:
            pass

    # ---------------- command handling ----------------

    def _process_command(self, command: str) -> Optional[str]:
        """
        Returns the text STAR SARA should speak, or None if the assistant
        should shut down. WHY the outer try/except in the caller matters:
        previously an exception inside intent handling (e.g. a bad Groq
        response, a note-search bug) meant the turn ended with NOTHING
        spoken and nothing logged — the owner just saw silence and had no
        idea whether STAR SARA heard them at all.
        """
        command = command.lower().strip()

        if not command:
            return f"I did not hear your command, {OWNER_ADDRESS}."

        intent = IntentClassifier.classify(command)

        if intent == "shutdown":
            self.shutdown_requested.emit()
            self._running = False
            return f"Goodbye {OWNER_ADDRESS}. Shutting down STAR SARA."

        payload = strip_phrases(command, IntentClassifier.trigger_phrases(intent))

        if intent == "remember":
            return self.ai.remember("user_note", payload)

        if intent == "add_task":
            self.ai.add_task(payload)
            return f"I added this task, {OWNER_ADDRESS}."

        if intent == "list_tasks":
            return self.ai.format_tasks()

        if intent == "tasks_due_today":
            return self.ai.format_due_today()

        if intent == "save_note":
            self.ai.save_note("Voice Note", payload)
            return f"I saved your note, {OWNER_ADDRESS}."

        if intent == "list_notes":
            return self.ai.format_notes()

        if intent == "search_notes":
            return self.ai.search_notes(payload)

        # default: free-form conversation, with full context/memory/history
        return self.ai.ask(command)

    # ---------------- main loop ----------------

    def run(self) -> None:
        try:
            self.state_changed.emit("loading")
            self.voice.load_model()

            self.state_changed.emit("speaking")
            greeting = self.ai.user_data.get("assistant", {}).get(
                "greeting", f"Good evening {OWNER_ADDRESS}. {ASSISTANT_NAME} is now online."
            )
            self.assistant_said.emit(greeting)
            self.voice.speak(greeting)

            self.state_changed.emit("idle")

            while self._running:
                try:
                    text = self.voice.listen(LISTEN_DURATION_WAKE)

                    if text:
                        print(f"Heard: {text}")

                    if self.voice.contains_wake_word(text):
                        self._handle_activation()

                except Exception as loop_error:
                    self.error_occurred.emit(str(loop_error))
                    traceback.print_exc()
                    time.sleep(1)

        except Exception as fatal_error:
            self.error_occurred.emit(str(fatal_error))
            traceback.print_exc()

    def _handle_activation(self) -> None:
        """
        One wake-word activation, followed by a loop of turns: the first
        turn waits up to LISTEN_DURATION_COMMAND for the owner to speak,
        every turn after that waits up to CONVERSATION_FOLLOWUP_TIMEOUT
        ("continuous conversation mode" — no need to repeat the wake word).
        The loop ends as soon as a turn times out with nothing heard.
        """
        self.state_changed.emit("listening")
        prompt = f"Yes {OWNER_ADDRESS}, how can I help you?"
        self.assistant_said.emit(prompt)
        self.voice.speak(prompt)

        is_first_turn = True

        while self._running:
            timeout = LISTEN_DURATION_COMMAND if is_first_turn else CONVERSATION_FOLLOWUP_TIMEOUT
            is_first_turn = False

            command = self.voice.listen(timeout)
            if not self._running:
                return
            if not command:
                break  # nothing heard within the window — back to idle

            command = self.voice.remove_wake_word(command)
            if not command:
                break

            self.owner_said.emit(command)
            self.state_changed.emit("processing")
            try:
                response = self._process_command(command)
            except Exception as error:
                traceback.print_exc()
                self.error_occurred.emit(str(error))
                response = _apology("something went wrong handling that.")

            if not self._running:
                return
            if not response:
                break  # e.g. shutdown intent already handled inside _process_command

            self.state_changed.emit("speaking")
            self.assistant_said.emit(response)
            self.voice.speak(response)

            self.state_changed.emit("listening")
            # loop continues — next iteration listens for a follow-up turn

        self.state_changed.emit("idle")


# ==============================================================================
# ASSISTANT STATE PRESENTATION
# ==============================================================================

# One row per assistant state instead of five parallel dicts keyed by the same
# state names (core colour, animation speed, wave amplitude, window title,
# subtitle) — adding or renaming a state now touches a single place.
STATE_PRESENTATION = {
    "loading": {
        "color": QColor(90, 90, 140),
        "speed": 0.6,
        "wave": 2.0,
        "title": "STAR SARA LOADING",
        "subtitle": "Warming up voice systems...",
    },
    "idle": {
        "color": QColor(0, 190, 255),
        "speed": 0.8,
        "wave": 3.0,
        "title": "STAR SARA ONLINE",
        "subtitle": "Waiting for command...",
    },
    "listening": {
        "color": QColor(0, 255, 220),
        "speed": 2.2,
        "wave": 9.0,
        "title": "STAR SARA ACTIVE",
        "subtitle": "Listening...",
    },
    "processing": {
        "color": QColor(170, 60, 255),
        "speed": 3.4,
        "wave": 14.0,
        "title": "STAR SARA THINKING",
        "subtitle": "Thinking...",
    },
    "speaking": {
        "color": QColor(80, 220, 255),
        "speed": 2.6,
        "wave": 11.0,
        "title": "STAR SARA RESPONDING",
        "subtitle": "Responding...",
    },
}


def state_attribute(state: str, attribute: str, default):
    return STATE_PRESENTATION.get(state, {}).get(attribute, default)


# ==============================================================================
# STAR CORE WIDGET — the glowing reactor animation (UNCHANGED from v2)
# ==============================================================================

class StarCoreWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 360)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        self.state = "loading"
        self.angle1 = 0.0
        self.angle2 = 0.0
        self.angle3 = 0.0
        self.pulse_phase = 0.0

        self.particles = []
        for _ in range(40):
            self.particles.append({
                "angle": random.uniform(0, 360),
                "radius": random.uniform(0.35, 0.98),
                "speed": random.uniform(0.3, 1.4),
                "size": random.uniform(1.5, 3.5),
            })

        self.wave_amplitude = 0.0
        self.target_wave_amplitude = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def set_state(self, state: str) -> None:
        if state in STATE_PRESENTATION:
            self.state = state
            self.target_wave_amplitude = state_attribute(state, "wave", 3.0)

    def _tick(self) -> None:
        speed = state_attribute(self.state, "speed", 1.0)

        self.angle1 = (self.angle1 + speed * 1.4) % 360
        self.angle2 = (self.angle2 - speed * 0.9) % 360
        self.angle3 = (self.angle3 + speed * 0.5) % 360
        self.pulse_phase = (self.pulse_phase + speed * 3.0) % 360

        self.wave_amplitude += (self.target_wave_amplitude - self.wave_amplitude) * 0.08

        for particle in self.particles:
            particle["angle"] = (particle["angle"] + particle["speed"] * speed) % 360

        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        width = self.width()
        height = self.height()
        center = QPointF(width / 2, height / 2)
        base_radius = min(width, height) * 0.36

        color = state_attribute(self.state, "color", QColor(0, 190, 255))
        pulse = (math.sin(math.radians(self.pulse_phase)) + 1) / 2

        glow_radius = base_radius * (1.55 + 0.12 * pulse)
        gradient = QRadialGradient(center, glow_radius)
        glow_color = QColor(color)
        glow_color.setAlpha(140)
        transparent = QColor(color)
        transparent.setAlpha(0)
        gradient.setColorAt(0.0, glow_color)
        gradient.setColorAt(1.0, transparent)
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, glow_radius, glow_radius)

        ring_specs = [
            (base_radius * 1.30, self.angle1, 3, 260),
            (base_radius * 1.12, self.angle2, 2, 200),
            (base_radius * 0.92, self.angle3, 2, 300),
        ]

        for radius, angle, width_px, span in ring_specs:
            pen = QPen(color)
            pen.setWidthF(width_px)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            rect = QRectF(
                center.x() - radius, center.y() - radius,
                radius * 2, radius * 2
            )
            painter.drawArc(rect, int(angle * 16), int(span * 16))

        wave_pen = QPen(color)
        wave_pen.setWidthF(1.5)
        painter.setPen(wave_pen)
        path = QPainterPath()
        points = 90
        wave_radius = base_radius * 0.78
        for i in range(points + 1):
            theta = (i / points) * 2 * math.pi
            wobble = self.wave_amplitude * math.sin(theta * 6 + math.radians(self.pulse_phase * 2))
            r = wave_radius + wobble
            x, y = polar_point(center.x(), center.y(), r, theta)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        painter.drawPath(path)

        painter.setPen(Qt.NoPen)
        for particle in self.particles:
            r = base_radius * particle["radius"]
            theta = math.radians(particle["angle"])
            x, y = polar_point(center.x(), center.y(), r, theta)
            dot_color = QColor(color)
            dot_color.setAlpha(180)
            painter.setBrush(QBrush(dot_color))
            size = particle["size"]
            painter.drawEllipse(QPointF(x, y), size, size)

        core_radius = base_radius * (0.42 + 0.03 * pulse)
        core_gradient = QRadialGradient(center, core_radius)
        core_gradient.setColorAt(0.0, QColor(10, 14, 22))
        core_bright = QColor(color)
        core_bright.setAlpha(230)
        core_gradient.setColorAt(0.75, core_bright)
        core_gradient.setColorAt(1.0, QColor(10, 14, 22))
        painter.setBrush(QBrush(core_gradient))
        pen = QPen(color)
        pen.setWidthF(2)
        painter.setPen(pen)
        painter.drawEllipse(center, core_radius, core_radius)

        star_path = self._build_star_path(center, core_radius * 0.5)
        painter.setPen(Qt.NoPen)
        star_color = QColor(255, 255, 255)
        star_color.setAlpha(220)
        painter.setBrush(QBrush(star_color))
        painter.drawPath(star_path)

        painter.end()

    @staticmethod
    def _build_star_path(center: QPointF, radius: float) -> QPainterPath:
        points = QPolygonF()
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r = radius if i % 2 == 0 else radius * 0.42
            x, y = polar_point(center.x(), center.y(), r, angle)
            points.append(QPointF(x, y))
        path = QPainterPath()
        path.addPolygon(points)
        path.closeSubpath()
        return path


# ==============================================================================
# INFO PANEL — futuristic side status panel (UNCHANGED from v2)
# ==============================================================================

class InfoPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("infoPanel")
        self.setFixedWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(14)

        title = QLabel("STAR SARA")
        title.setObjectName("panelTitle")
        layout.addWidget(title)

        subtitle = QLabel("Smart AI Response Assistant")
        subtitle.setObjectName("panelSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        self.owner_label = self._add_row(layout, "Owner", OWNER_ADDRESS)
        self.engine_label = self._add_row(layout, "AI Engine", "Groq Llama 3.3")
        self.voice_label = self._add_row(layout, "Voice", "Female Neural Voice")
        self.status_label = self._add_row(layout, "Status", "Initializing...")
        self.memory_label = self._add_row(layout, "Memory", "Connected")

        layout.addStretch()

    def _add_row(self, layout: QVBoxLayout, label_text: str, value_text: str) -> QLabel:
        row = QFrame()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(2)

        caption = QLabel(label_text.upper())
        caption.setObjectName("panelCaption")

        value = QLabel(value_text)
        value.setObjectName("panelValue")
        value.setWordWrap(True)

        row_layout.addWidget(caption)
        row_layout.addWidget(value)
        layout.addWidget(row)

        return value

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


# ==============================================================================
# MAIN WINDOW (UNCHANGED from v2)
# ==============================================================================

class FuturisticGUI(QMainWindow):

    def __init__(self, core: "StarSaraCore"):
        super().__init__()
        self.core = core

        self.setWindowTitle("STAR SARA — Personal AI Executive Assistant")
        self.resize(1100, 680)
        self.setStyleSheet(self._stylesheet())

        central = QWidget()
        central.setObjectName("rootWidget")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        core_column = QWidget()
        core_layout = QVBoxLayout(core_column)
        core_layout.setContentsMargins(30, 30, 30, 30)
        core_layout.setSpacing(16)
        core_layout.setAlignment(Qt.AlignHCenter)

        self.title_label = QLabel(state_attribute("loading", "title", "STAR SARA"))
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.subtitle_label = QLabel(state_attribute("loading", "subtitle", ""))
        self.subtitle_label.setObjectName("subtitleLabel")
        self.subtitle_label.setAlignment(Qt.AlignCenter)

        self.star_core = StarCoreWidget()

        self.log_view = QTextEdit()
        self.log_view.setObjectName("logView")
        self.log_view.setReadOnly(True)
        self.log_view.setFixedHeight(160)

        core_layout.addWidget(self.title_label)
        core_layout.addWidget(self.subtitle_label)
        core_layout.addWidget(self.star_core, stretch=1)
        core_layout.addWidget(self.log_view)

        self.info_panel = InfoPanel()

        main_layout.addWidget(core_column, stretch=1)
        main_layout.addWidget(self.info_panel)

    @Slot(str)
    def on_state_changed(self, state: str) -> None:
        self.star_core.set_state(state)
        self.title_label.setText(state_attribute(state, "title", "STAR SARA"))
        self.subtitle_label.setText(state_attribute(state, "subtitle", ""))
        self.info_panel.set_status(state_attribute(state, "title", "Online"))

    @Slot(str)
    def on_owner_said(self, text: str) -> None:
        self._append_log(f"{OWNER_ADDRESS}: {text}")

    @Slot(str)
    def on_assistant_said(self, text: str) -> None:
        self._append_log(f"{ASSISTANT_NAME}: {text}")

    @Slot(str)
    def on_error(self, message: str) -> None:
        self._append_log(f"[SYSTEM] {message}")

    def _append_log(self, line: str) -> None:
        self.log_view.append(line)
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event) -> None:
        self.core.shutdown()
        event.accept()

    @staticmethod
    def _stylesheet() -> str:
        return """
            #rootWidget {
                background-color: #05070d;
            }
            #titleLabel {
                color: #8fe9ff;
                font-family: 'Segoe UI', 'Consolas', monospace;
                font-size: 26px;
                font-weight: 600;
                letter-spacing: 3px;
            }
            #subtitleLabel {
                color: #5c7a8c;
                font-family: 'Segoe UI', 'Consolas', monospace;
                font-size: 13px;
                letter-spacing: 2px;
            }
            #logView {
                background-color: rgba(10, 16, 28, 0.85);
                border: 1px solid rgba(0, 190, 255, 0.25);
                border-radius: 8px;
                color: #9fe8ff;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 8px;
            }
            #infoPanel {
                background-color: rgba(8, 12, 22, 0.95);
                border-left: 1px solid rgba(0, 190, 255, 0.25);
            }
            #panelTitle {
                color: #ffffff;
                font-family: 'Segoe UI', monospace;
                font-size: 20px;
                font-weight: 700;
                letter-spacing: 2px;
            }
            #panelSubtitle {
                color: #5c7a8c;
                font-size: 11px;
            }
            #panelCaption {
                color: #3f5b6b;
                font-size: 10px;
                letter-spacing: 1px;
            }
            #panelValue {
                color: #b7f2ff;
                font-size: 14px;
                font-weight: 500;
            }
        """


# ==============================================================================
# TOP LEVEL CONTROLLER (UNCHANGED from v2)
# ==============================================================================

class StarSaraCore:
    def __init__(self):
        self.ai_engine = AIEngine()
        self.voice_engine = VoiceEngine()
        self.worker = AssistantWorker(self.ai_engine, self.voice_engine)
        self.gui: Optional[FuturisticGUI] = None

    def attach_gui(self, gui: FuturisticGUI) -> None:
        self.gui = gui
        self.worker.state_changed.connect(gui.on_state_changed)
        self.worker.owner_said.connect(gui.on_owner_said)
        self.worker.assistant_said.connect(gui.on_assistant_said)
        self.worker.error_occurred.connect(gui.on_error)
        self.worker.shutdown_requested.connect(self._on_shutdown_requested)

    def start(self) -> None:
        self.worker.start()

    def shutdown(self) -> None:
        self.worker.stop()
        self.worker.wait(2000)

    def _on_shutdown_requested(self) -> None:
        if self.gui:
            QTimer.singleShot(1500, self.gui.close)


# ==============================================================================
# PROGRAM START
# ==============================================================================

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("STAR SARA")

    core = StarSaraCore()
    gui = FuturisticGUI(core)
    core.attach_gui(gui)

    gui.show()
    core.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()