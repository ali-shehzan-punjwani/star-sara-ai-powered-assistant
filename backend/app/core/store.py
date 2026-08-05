"""Small JSON-file persistence layer with fuzzy text helpers."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

try:
    from rapidfuzz import fuzz

    _RAPIDFUZZ = True
except ImportError:  # pragma: no cover - optional dependency
    _RAPIDFUZZ = False


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        save_json(path, default)
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def similarity(a: str, b: str) -> float:
    """Fuzzy similarity on a 0-100 scale."""
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if _RAPIDFUZZ:
        return float(fuzz.partial_ratio(a, b))
    return SequenceMatcher(None, a, b).ratio() * 100
