# STAR SARA — AI Executive Assistant Platform

**Smart AI Response Assistant** · by **STAR Technologies** · Founder & CEO **Ali Shehzan Punjwani**

A realtime, browser-native voice assistant: a Jarvis-style holographic operations
dashboard on top of a streaming speech pipeline. Replaces the PySide6 desktop
build (v3.0) with a Next.js + FastAPI product — the old app is preserved under
[`legacy/`](legacy/).

---

## Architecture

```
Browser mic ──AudioWorklet PCM16/16 kHz──▶ WebSocket ─┐
                                                      ▼
                                        Silero VAD (WebRTC fallback)
                                                      ▼
                                        Wake word (openWakeWord / Porcupine
                                          / fuzzy transcript fallback)
                                                      ▼
                                        faster-whisper (GPU → int8 CPU)
                                                      ▼
                                        Groq Llama 3.3 70B — token streaming
                                              │ tokens          │ sentences
                                              ▼                 ▼
                                     WebSocket to UI     streaming Edge TTS
                                                                ▼
                                                    mp3 chunks ──▶ browser playback
```

Nothing in the chain waits for the stage before it to finish: the first TTS
sentence is synthesized while Groq is still generating, and Whisper starts the
moment VAD sees trailing silence rather than after a fixed recording window.

| Stage | Target | How it is achieved |
| --- | --- | --- |
| Wake word | < 500 ms | frame-level detector, no transcription in the hot path |
| Speech recognition | < 2 s | `base.en`/`small.en` CTranslate2, int8 CPU or fp16 GPU, greedy decode in fast mode |
| First AI token | < 1 s | Groq streaming API, intents answered locally with zero network hops |
| Voice start | < 2 s | sentence-level TTS streaming, playback begins on chunk one |

Measured latency for every turn is streamed back as a `metrics` event and shown
live in the **AI Engine** card.

## Features

- **Holographic AI core** with four states — idle sphere, listening pulses,
  thinking neural mesh, responding voice wave — plus live input/output waveforms.
- **Operations dashboard**: assistant status, memory count, upcoming tasks,
  CPU/RAM/battery/network, engine + latency telemetry.
- **Chat**: streaming markdown with code highlighting, timestamps, voice-message
  markers, spring animations.
- **Continuous conversation**: after a reply, follow-ups need no wake word for
  12 s (or permanently, via the toggle).
- **Barge-in**: talking over the assistant stops playback instantly.
- **Executive memory**: fuzzy-deduplicated facts with importance and decay,
  tasks, and notes — ported from the desktop build and exposed over REST.
- **Selectable accuracy**: fast / balanced / accurate map to whisper
  `base.en` / `small.en` / `medium.en`.

## Quick start

```bash
# 1. Backend
cd backend
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
.venv/bin/python scripts/download_models.py   # optional: wake-word models
.venv/bin/uvicorn app.main:app --reload --port 8000

# 2. Frontend
cd frontend
npm install
cp .env.example .env.local    # points at http://localhost:8000
npm run dev                   # http://localhost:3000
```

Click **Activate microphone**, allow access, then say *"STAR SARA, explain AWS
IAM"*. Text chat works without a microphone.

## Configuration

Backend (`backend/.env`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | required for the LLM; without it the UI still runs and says the brain is offline |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model id |
| `ACCURACY_MODE` | `fast` | `fast` / `balanced` / `accurate` |
| `WHISPER_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `WHISPER_COMPUTE_TYPE` | `auto` | `float16` on GPU, `int8` on CPU |
| `TTS_VOICE` | `en-US-AriaNeural` | Edge TTS voice |
| `WAKE_WORDS` | `["star sara","sara"]` | JSON list |
| `PORCUPINE_ACCESS_KEY` | — | enables the Porcupine wake engine |
| `OPENWAKEWORD_MODEL_PATH` | — | path to a custom trained "STAR SARA" model |
| `FOLLOWUP_WINDOW_SECONDS` | `12` | wake-word-free follow-up window |

Frontend (`frontend/.env.local`): `NEXT_PUBLIC_API_BASE=http://localhost:8000`.

### Wake word accuracy

openWakeWord ships no "STAR SARA" model, so the acoustic engine is only used
when you point `OPENWAKEWORD_MODEL_PATH` at a custom model (or set
`OPENWAKEWORD_USE_BUNDLED=1` to trial the bundled phrases). Otherwise wake
detection falls back to fuzzy-matching the transcript, which is accurate but
adds one transcription of latency. Training a custom model takes minutes with
the openWakeWord notebook and is the single biggest wake-latency win.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | liveness |
| `GET` | `/api/status` | assistant, owner, engine and counts |
| `GET` | `/api/system` | CPU / RAM / battery / network |
| `GET`,`POST` | `/api/memories` | list, create |
| `DELETE` | `/api/memories/{key}` | forget |
| `GET`,`POST` | `/api/tasks` | list, create |
| `POST` | `/api/tasks/{id}/complete` | complete |
| `GET`,`POST` | `/api/notes` | list, create |
| `POST` | `/api/conversation/reset` | clear short-term history |
| `WS` | `/ws/voice` | binary PCM16 in, JSON events out |

## Tests

```bash
cd backend && .venv/bin/python -m pytest       # VAD, wake word, intents, turn loop, API
cd frontend && npm run lint && npx tsc --noEmit && npm run build
```
