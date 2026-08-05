"""REST surface backing the dashboard cards."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..config import WHISPER_MODELS, settings
from ..services import system
from ..services.llm import brain
from ..services.memory import store
from ..services.stt import recognizer
from ..services.wakeword import WakeWordDetector

router = APIRouter(prefix="/api")
_wake_probe = WakeWordDetector()


class MemoryIn(BaseModel):
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    importance: int = 3


class TaskIn(BaseModel):
    task: str = Field(min_length=1)
    priority: str = "normal"
    due: Optional[str] = None


class NoteIn(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


@router.get("/status")
def status() -> dict[str, Any]:
    return {
        "assistant": settings.assistant_name,
        "online": True,
        "owner": {
            "name": settings.owner_name,
            "title": settings.owner_title,
            "company": settings.company,
            "address_as": settings.owner_address,
        },
        "engine": {
            "llm": settings.groq_model,
            "llm_online": brain.online,
            "stt": settings.whisper_model_name,
            "stt_available": recognizer.available,
            "stt_device": recognizer.device,
            "stt_compute": recognizer.compute_type,
            "wake_word": _wake_probe.backend_name,
            "tts_voice": settings.tts_voice,
            "accuracy_mode": settings.accuracy_mode,
            "accuracy_modes": WHISPER_MODELS,
        },
        "counts": {
            "memories": store.fact_count,
            "pending_tasks": len(store.pending_tasks()),
            "due_today": len(store.tasks_due_today()),
            "notes": len(store.notes.get("notes", [])),
        },
    }


@router.get("/system")
def system_stats() -> dict[str, Any]:
    return system.snapshot()


@router.get("/memories")
def list_memories() -> dict[str, Any]:
    facts = sorted(
        store.memory.get("facts", []), key=lambda f: f.get("last_accessed", ""), reverse=True
    )
    return {"count": len(facts), "facts": facts}


@router.post("/memories", status_code=201)
def create_memory(payload: MemoryIn) -> dict[str, Any]:
    return store.remember(payload.key, payload.value, payload.importance)


@router.delete("/memories/{key}")
def delete_memory(key: str) -> dict[str, bool]:
    if not store.forget(key):
        raise HTTPException(status_code=404, detail="memory not found")
    return {"deleted": True}


@router.get("/tasks")
def list_tasks() -> dict[str, Any]:
    return {
        "tasks": store.tasks.get("tasks", []),
        "pending": len(store.pending_tasks()),
        "due_today": store.tasks_due_today(),
    }


@router.post("/tasks", status_code=201)
def create_task(payload: TaskIn) -> dict[str, Any]:
    return store.add_task(payload.task, payload.priority, payload.due)


@router.post("/tasks/{task_id}/complete")
def complete_task(task_id: str) -> dict[str, bool]:
    if not store.complete_task(task_id):
        raise HTTPException(status_code=404, detail="task not found")
    return {"completed": True}


@router.get("/notes")
def list_notes() -> dict[str, Any]:
    return {"notes": store.notes.get("notes", [])}


@router.post("/notes", status_code=201)
def create_note(payload: NoteIn) -> dict[str, Any]:
    return store.save_note(payload.title, payload.content)


@router.post("/conversation/reset")
def reset_conversation() -> dict[str, bool]:
    brain.reset()
    return {"reset": True}
