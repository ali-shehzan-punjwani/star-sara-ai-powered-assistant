"""Long-term memory, tasks and notes for STAR SARA.

Ported from the desktop v3 AIEngine, minus the Groq/GUI coupling: the store is
now a standalone service the API and the voice pipeline both read from.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Optional

from ..config import DATA_DIR, settings
from ..core.store import load_json, save_json, similarity

USER_FILE = DATA_DIR / "user_data.json"
MEMORY_FILE = DATA_DIR / "memory.json"
TASKS_FILE = DATA_DIR / "tasks.json"
NOTES_FILE = DATA_DIR / "notes.json"

SENSITIVE_PROFILE_FIELDS = {"date_of_birth", "religion", "contact", "family", "phone", "email"}

DEFAULT_PROFILE: dict[str, Any] = {
    "name": settings.owner_name,
    "title": settings.owner_title,
    "company": settings.company,
    "role": "AI & Cloud Engineering",
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self) -> None:
        self.profile = load_json(USER_FILE, DEFAULT_PROFILE)
        self.memory = load_json(MEMORY_FILE, {"facts": []})
        self.tasks = load_json(TASKS_FILE, {"tasks": []})
        self.notes = load_json(NOTES_FILE, {"notes": []})
        self._migrate_facts()
        self._decay()

    # ------------------------------------------------------------------ memory

    def _migrate_facts(self) -> None:
        for fact in self.memory.get("facts", []):
            fact.setdefault("importance", 3)
            fact.setdefault("created_at", _now())
            fact.setdefault("last_accessed", fact["created_at"])
            fact.setdefault("access_count", 0)

    def _decay(self) -> None:
        """Drop stale, low-importance facts so context stays relevant."""
        cutoff = datetime.now() - timedelta(days=settings.memory_decay_days)
        kept = []
        for fact in self.memory.get("facts", []):
            try:
                last = datetime.fromisoformat(fact["last_accessed"])
            except (KeyError, ValueError):
                last = datetime.now()
            stale = last < cutoff and fact.get("access_count", 0) < 1
            if stale and fact.get("importance", 3) <= 2:
                continue
            kept.append(fact)
        if len(kept) != len(self.memory.get("facts", [])):
            self.memory["facts"] = kept
            save_json(MEMORY_FILE, self.memory)

    def remember(self, key: str, value: str, importance: int = 3) -> dict[str, Any]:
        facts = self.memory.setdefault("facts", [])
        for fact in facts:
            if similarity(fact["key"], key) >= settings.memory_dedupe_threshold:
                fact.update(value=value, importance=importance, last_accessed=_now())
                save_json(MEMORY_FILE, self.memory)
                return fact

        fact = {
            "key": key.strip(),
            "value": value.strip(),
            "importance": max(1, min(5, importance)),
            "created_at": _now(),
            "last_accessed": _now(),
            "access_count": 0,
        }
        facts.append(fact)
        if len(facts) > settings.memory_max_facts:
            facts.sort(key=lambda f: (f["importance"], f["last_accessed"]))
            del facts[: len(facts) - settings.memory_max_facts]
        save_json(MEMORY_FILE, self.memory)
        return fact

    def forget(self, key: str) -> bool:
        facts = self.memory.get("facts", [])
        remaining = [f for f in facts if f["key"].lower() != key.lower()]
        if len(remaining) == len(facts):
            return False
        self.memory["facts"] = remaining
        save_json(MEMORY_FILE, self.memory)
        return True

    def relevant_memories(self, query: str, top_k: Optional[int] = None) -> list[dict[str, Any]]:
        top_k = top_k or settings.memory_top_k
        scored: list[tuple[float, dict[str, Any]]] = []
        for fact in self.memory.get("facts", []):
            score = max(similarity(query, fact["key"]), similarity(query, fact["value"]))
            scored.append((score + fact.get("importance", 3) * 2, fact))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        selected = [fact for _, fact in scored[:top_k]]
        for fact in selected:
            fact["last_accessed"] = _now()
            fact["access_count"] = fact.get("access_count", 0) + 1
        if selected:
            save_json(MEMORY_FILE, self.memory)
        return selected

    @property
    def fact_count(self) -> int:
        return len(self.memory.get("facts", []))

    # ------------------------------------------------------------------- tasks

    def add_task(
        self, text: str, priority: str = "normal", due: Optional[str] = None
    ) -> dict[str, Any]:
        task = {
            "id": f"task-{len(self.tasks.get('tasks', [])) + 1}-{int(datetime.now().timestamp())}",
            "task": text.strip(),
            "priority": priority,
            "due": due,
            "status": "pending",
            "created_at": _now(),
        }
        self.tasks.setdefault("tasks", []).append(task)
        save_json(TASKS_FILE, self.tasks)
        return task

    def pending_tasks(self) -> list[dict[str, Any]]:
        return [t for t in self.tasks.get("tasks", []) if t.get("status") == "pending"]

    def tasks_due_today(self) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        return [t for t in self.pending_tasks() if (t.get("due") or "").startswith(today)]

    def complete_task(self, task_id: str) -> bool:
        for task in self.tasks.get("tasks", []):
            if task["id"] == task_id:
                task["status"] = "done"
                task["completed_at"] = _now()
                save_json(TASKS_FILE, self.tasks)
                return True
        return False

    # ------------------------------------------------------------------- notes

    def save_note(self, title: str, content: str) -> dict[str, Any]:
        note = {
            "id": f"note-{int(datetime.now().timestamp())}",
            "title": title.strip(),
            "content": content.strip(),
            "created_at": _now(),
        }
        self.notes.setdefault("notes", []).append(note)
        save_json(NOTES_FILE, self.notes)
        return note

    def search_notes(self, query: str) -> list[dict[str, Any]]:
        notes = self.notes.get("notes", [])
        matches = [
            note
            for note in notes
            if query.lower() in f"{note['title']} {note['content']}".lower()
            or similarity(query, note["title"]) >= 80
        ]
        return matches or notes[-5:]

    # ----------------------------------------------------------------- context

    def llm_safe_profile(self) -> dict[str, Any]:
        return {k: v for k, v in self.profile.items() if k not in SENSITIVE_PROFILE_FIELDS}


store = MemoryStore()
