# STAR SARA v2.0 — Setup Guide (Windows, Python 3.12.10)

## 1. Project files
Put these in one folder:
- `main.py`
- `requirements.txt`
- `.env` (you create this — see step 3)

The first run auto-creates `user_data.json`, `memory.json`, `tasks.json`, `notes.json` next to `main.py`.

## 2. Install system dependency: FFmpeg
Whisper needs FFmpeg on your PATH.
- Easiest: `winget install ffmpeg` (PowerShell), then restart the terminal.
- Or download a build from ffmpeg.org and add its `bin` folder to PATH.

## 3. Create your `.env` file
In the project folder, create a file named `.env` with:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at console.groq.com. If this key is missing, the app still opens and runs voice/GUI features, but AI replies will say the brain is offline.

## 4. Create a virtual environment and install packages
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Notes:
- `openai-whisper` + `torch` are the heaviest installs (a few GB) — this is expected.
- If `torch` install is slow/fails, install the CPU wheel directly first:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
  then re-run `pip install -r requirements.txt`.

## 5. Run it
```powershell
python main.py
```

A dark window opens with a glowing STAR CORE and a side status panel.
The whisper model loads on first launch (shows "STAR SARA LOADING") — this
can take 10–30 seconds depending on your machine, then you'll hear the
greeting and the core settles into idle "breathing" mode.

## 6. Talking to it
- Say **"STAR SARA"** or just **"SARA"** — the core brightens and speeds up (LISTENING).
- After it replies "Yes Shehzan Sir, how can I help you?", speak your command.
- Built-in voice commands (handled without calling the AI):
  - `remember <something>` — stores a fact in memory.json
  - `add task <something>` — adds a pending task
  - `my tasks` / `pending tasks` — reads back pending tasks
  - `save note <something>` — saves a voice note
  - `my notes` — reads back saved notes
  - `exit` / `quit` / `shutdown` / `stop` / `goodbye` — says goodbye and closes the app
- Anything else is sent to the Groq Llama model, with your profile, memory,
  tasks, and notes included as context.

## Troubleshooting
- **No sound / mic not detected**: check Windows sound settings default
  input/output devices; `sounddevice` uses whatever Windows has set as default.
- **"GROQ_API_KEY missing" in console**: confirm `.env` is in the same folder
  as `main.py` and has no quotes around the key.
- **GUI freezes**: all voice/AI work runs on a background `QThread`
  (`AssistantWorker`), so the animation loop should keep running even while
  listening/thinking/speaking — if it still freezes, check the console for
  a Python traceback.
