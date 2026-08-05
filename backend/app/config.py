"""Runtime configuration for the STAR SARA backend."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

AccuracyMode = Literal["fast", "balanced", "accurate"]

# Whisper model per accuracy tier. `fast` targets sub-second transcription on
# CPU, `accurate` trades ~3x latency for noticeably better proper-noun recall.
WHISPER_MODELS: dict[str, str] = {
    "fast": "base.en",
    "balanced": "small.en",
    "accurate": "medium.en",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.5
    groq_max_tokens: int = 500

    assistant_name: str = "STAR SARA"
    owner_address: str = "Shehzan Sir"
    owner_name: str = "Ali Shehzan Punjwani"
    owner_title: str = "Founder & CEO"
    company: str = "STAR Technologies"

    wake_words: list[str] = ["star sara", "sara"]
    wake_word_fuzzy_threshold: int = 90
    wake_word_cooldown_seconds: float = 2.5

    accuracy_mode: AccuracyMode = "fast"
    whisper_device: str = "auto"  # auto | cuda | cpu
    whisper_compute_type: str = "auto"  # auto -> float16 on GPU, int8 on CPU

    sample_rate: int = 16000
    vad_frame_ms: int = 30
    vad_speech_threshold: float = 0.5
    vad_silence_timeout_ms: int = 600
    vad_min_speech_ms: int = 250
    vad_max_utterance_seconds: float = 20.0

    tts_voice: str = "en-US-AriaNeural"
    tts_rate: str = "+8%"
    tts_pitch: str = "+0Hz"

    # Seconds of silence after a reply during which the next utterance needs no
    # wake word.
    followup_window_seconds: float = 12.0
    history_turns: int = 6

    memory_max_facts: int = 300
    memory_dedupe_threshold: int = 88
    memory_top_k: int = 8
    memory_decay_days: int = 30

    cors_origins: list[str] = ["http://localhost:3000"]

    @property
    def whisper_model_name(self) -> str:
        return WHISPER_MODELS[self.accuracy_mode]


settings = Settings()
DATA_DIR.mkdir(parents=True, exist_ok=True)
