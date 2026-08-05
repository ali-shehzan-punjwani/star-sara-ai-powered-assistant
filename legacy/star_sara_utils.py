"""
==============================================================================
STAR SARA — shared utilities
==============================================================================

Small, dependency-light helpers that were previously duplicated inline across
star_sara_v3.py (JSON persistence, fuzzy text similarity, timestamps, error
logging, numbered speech lists, phrase stripping, polar geometry).
==============================================================================
"""

import json
import math
import os
from datetime import datetime
from typing import Any, Callable, Iterable, Sequence, Tuple

try:
    from rapidfuzz import fuzz as _fuzz
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    import difflib
    RAPIDFUZZ_AVAILABLE = False


# ==============================================================================
# LOGGING
# ==============================================================================

def log_error(context: str, error: Any) -> None:
    print(f"[ERROR] {context}: {error}")


def log_warning(context: str, error: Any) -> None:
    print(f"[WARN] {context}: {error}")


# ==============================================================================
# TIME
# ==============================================================================

def now_iso() -> str:
    """Timestamp used by every persisted record (facts, tasks, notes)."""
    return datetime.now().isoformat()


# ==============================================================================
# JSON PERSISTENCE
# ==============================================================================

def save_json(file_path: str, data: dict) -> None:
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except Exception as error:
        log_error(f"Saving JSON ({file_path})", error)


def load_json(file_path: str, default: dict) -> dict:
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        save_json(file_path, default)
        return default
    except Exception as error:
        log_error(f"Loading JSON ({file_path})", error)
        return default


def load_json_collection(file_path: str, key: str) -> dict:
    """
    Loads a `{key: [...]}` document, tolerating files written by older
    versions (or hand-edited / emptied ones) that are missing the key or are
    not even a dict — every caller used to repeat this same guard before
    indexing the collection.
    """
    data = load_json(file_path, {key: []})
    if not isinstance(data, dict):
        data = {key: []}
    data.setdefault(key, [])
    return data


# ==============================================================================
# TEXT
# ==============================================================================

def text_similarity(a: str, b: str) -> float:
    """
    Unified fuzzy-similarity helper (0-100) that uses rapidfuzz when available
    and falls back to stdlib difflib otherwise, so every fuzzy-matching
    feature (wake word, memory dedupe, intent scoring, note search) keeps
    working with zero extra installs, just less accurately.
    """
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return 0.0
    if RAPIDFUZZ_AVAILABLE:
        return _fuzz.partial_ratio(a, b)
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def strip_phrases(text: str, phrases: Iterable[str]) -> str:
    """
    Removes the first occurrence of each trigger phrase from a command, e.g.
    "add task call the bank" -> "call the bank".
    """
    for phrase in phrases:
        text = text.replace(phrase, "", 1)
    return text.strip()


def format_numbered_list(items: Sequence[Any], header: str, label: str,
                         render: Callable[[Any], str]) -> str:
    """
    Builds the "You have N things. Thing 1: ... Thing 2: ..." speech string
    shared by the task and note read-back responses.
    """
    response = header
    for number, item in enumerate(items, start=1):
        response += f"{label} {number}: {render(item)}. "
    return response


# ==============================================================================
# GEOMETRY (GUI rendering)
# ==============================================================================

def polar_point(center_x: float, center_y: float, radius: float,
                theta_radians: float) -> Tuple[float, float]:
    """Cartesian coordinates of a point at `radius`/`theta` around a center."""
    return (center_x + radius * math.cos(theta_radians),
            center_y + radius * math.sin(theta_radians))
