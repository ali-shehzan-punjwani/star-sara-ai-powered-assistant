# Legacy desktop build (v3.0)

The original PySide6 desktop assistant, kept for reference and for anyone
running STAR SARA without a browser. It is no longer the primary product — see
the repository README for the Next.js + FastAPI platform that replaces it.

```bash
pip install -r requirements.txt
cp .env.example .env
python star_sara_v3.py
```

## Local data files

`user_data.json`, `memory.json`, `notes.json` and `tasks.json` hold personal
data and are git-ignored. On first run each one is created automatically from
its `*.example.json` template with owner-only (`0600`) permissions. Edit
`user_data.json` locally to personalize the assistant — never commit it.
