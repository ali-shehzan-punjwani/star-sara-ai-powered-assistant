---
name: testing-star-sara
description: How to run and test the STAR SARA web platform (Next.js dashboard + FastAPI voice backend) locally in a headless VM.
---

# Testing STAR SARA locally

## Bring up the stack
Backend (needs faster-whisper, silero-vad, onnxruntime, webrtcvad, openwakeword, edge-tts, groq, psutil, rapidfuzz):

```
cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000 > /tmp/backend.log 2>&1 &
```

First start downloads the faster-whisper `base.en` model from HuggingFace and warms it up — allow ~15 s and confirm with `curl -s localhost:8000/api/status` before touching the UI.

Frontend:

```
cd frontend && cp .env.example .env.local   # sets NEXT_PUBLIC_API_BASE=http://localhost:8000
npm install && npm run dev                  # :3000
```

## Environment limitations to expect
- **No `GROQ_API_KEY`**: free-form chat intentionally answers "My AI brain is offline right now, Shehzan Sir — set GROQ_API_KEY…". That is correct behaviour, not a bug. It also means markdown/code-fence rendering in chat bubbles cannot be exercised — mark it untested rather than guessing.
- **No microphone in the VM**: clicking "Activate microphone" is expected to surface `Requested device not found` in the header banner. Verify the error is surfaced and the page keeps working; do not fail the run.

## Testing the pipeline without a mic
The chat text box sends over the same `/ws/voice` WebSocket as voice, so local intents are fully testable by typing. Intent keywords live in `backend/app/services/intents.py`; useful probes:
- `add task <text>` → "Added to your list, Shehzan Sir: …", Tasks card increments
- `what tasks do I have` → enumerates pending tasks
- `remember that <fact>` → Memory card increments

Card counters come from `/api/status` + `/api/tasks` polled every **15 s** (`frontend/src/app/page.tsx`), and `/api/system` every **3 s**, so wait up to ~20 s before calling a counter assertion failed.

## Known trap: recognition mode reverts
`accuracy_mode` chosen in the UI is per-WebSocket-session only, and the 15 s `/api/status` poll used to overwrite the local `mode` state with the server's global default, silently snapping the highlighted button back to `fast` (fixed by a `modeSeeded` ref in `page.tsx`). When testing mode switching, always re-check the button state ~45 s after clicking (3 poll cycles), not just immediately — `/api/status` will still report `fast`, which is expected. Confirm the switch reached the backend with `grep 'Session config' <backend log>`.

## Testing with a live GROQ_API_KEY
Put the key in `backend/.env` (gitignored) and restart uvicorn; confirm with `curl -s localhost:8000/api/status` → `"llm_online": true`. Never echo the key.

- **Proving token streaming**: short answers finish in well under a second, so a screenshot will only ever show the completed bubble. Use a deliberately long prompt (e.g. "list and describe the twelve factors of the twelve-factor app methodology in detail") and screenshot immediately — you should catch partial text plus the `▍` caret and the core in `RESPONDING`.
- **Markdown / code blocks**: the system prompt (`backend/app/services/llm.py`) tells the model its reply is spoken aloud with no markdown, so normal questions never produce code fences. To exercise `rehypeHighlight`, force it: "output only a fenced markdown code block with triple backticks and the language python containing print('hello world'), nothing else".
- **TTS will not fire for typed messages**: `useVoiceSession.ts` sends `data: { speak: micActive }`, so with no mic the `1ST AUDIO` metric stays `—` — that is by design, not a bug. To verify the TTS leg without a mic, open a WebSocket to `ws://localhost:8000/ws/voice` and send `{"type":"text","text":"...","data":{"speak":true}}`, then count `audio` events and read `first_audio_ms` off the `metrics` event.
- **Local intents must not hit Groq**: after an `add task ...` turn the `1ST TOKEN` metric should read `0ms`; a non-zero value means the intent router missed and the LLM answered.
- The Tasks card lists at most 3 bullets (`StatCards.tsx` does `tasks.slice(0, 3)`) while the count is the true total — not a bug.
- The LLM answer legitimately weaves in stored memories (e.g. "your eu-west-1 region"), which is good extra evidence that the memory store feeds the system prompt.

## Devin Secrets Needed
- `GROQ_API_KEY` — only needed to test real LLM token streaming, TTS audio playback, and markdown/code rendering in chat.
