"""Keyword + fuzzy intent routing, so common commands never pay an LLM round trip."""

from __future__ import annotations

import re

from ..core.store import similarity

INTENT_KEYWORDS: dict[str, list[str]] = {
    "remember": ["remember that", "remember"],
    "add_task": ["add task", "new task", "add a task", "remind me to"],
    "list_tasks": ["my tasks", "pending tasks", "what tasks"],
    "tasks_due_today": ["due today", "what's due", "whats due"],
    "save_note": ["save note", "take a note", "new note"],
    "list_notes": ["my notes", "read my notes"],
    "search_notes": ["find my note", "search notes", "note about"],
    "stop": ["stop talking", "be quiet", "never mind"],
}

FUZZY_SCORE_THRESHOLD = 88
FUZZY_MAX_LENGTH_RATIO = 1.6


def classify(command: str) -> str:
    command = command.lower().strip()
    words = command.split()
    best_intent, best_score = "chat", 0.0

    for intent, phrases in INTENT_KEYWORDS.items():
        for phrase in phrases:
            if re.search(rf"\b{re.escape(phrase)}\b", command):
                return intent
            # Only fuzzy-match utterances of comparable length; otherwise a short
            # phrase scores highly against an unrelated long sentence.
            if len(words) > len(phrase.split()) * FUZZY_MAX_LENGTH_RATIO:
                continue
            score = similarity(phrase, command)
            if score > best_score:
                best_intent, best_score = intent, score

    return best_intent if best_score >= FUZZY_SCORE_THRESHOLD else "chat"


def strip_intent_prefix(command: str, intent: str) -> str:
    text = command.strip()
    for phrase in INTENT_KEYWORDS.get(intent, []):
        text = re.sub(rf"^\s*{re.escape(phrase)}\s*", "", text, flags=re.IGNORECASE)
    return text.strip(" ,.:")
