"""Tests for AssistantWorker._process_command intent dispatch.

These exercise the routing layer end-to-end (through IntentClassifier and the
AIEngine handlers) without any threading, audio, or GUI involvement.
"""

import pytest


@pytest.fixture
def worker(app_module, ai):
    voice = app_module.VoiceEngine()
    return app_module.AssistantWorker(ai, voice)


def test_empty_command_reports_nothing_heard(worker, app_module):
    reply = worker._process_command("   ")
    assert app_module.OWNER_ADDRESS in reply
    assert "did not hear" in reply.lower()


def test_add_task_command_creates_task(worker, ai):
    reply = worker._process_command("add task buy milk")

    assert "added this task" in reply.lower()
    pending = ai.get_pending_tasks()
    assert len(pending) == 1
    assert pending[0]["task"] == "buy milk"


def test_remember_command_stores_fact(worker, ai):
    reply = worker._process_command("remember that I like tea")

    assert "remember" in reply.lower()
    assert ai.memory["facts"][0]["value"] == "i like tea"


def test_save_note_command_stores_note(worker, ai):
    reply = worker._process_command("take a note buy bread tomorrow")

    assert "saved your note" in reply.lower()
    assert ai.notes["notes"][0]["content"] == "buy bread tomorrow"


def test_list_tasks_command(worker, ai):
    ai.add_task("existing task")
    reply = worker._process_command("show my tasks")
    assert "existing task" in reply


def test_search_notes_command(worker, ai):
    ai.save_note("Groceries", "buy milk and eggs")
    reply = worker._process_command("find my note about groceries")
    assert "Groceries" in reply


def test_shutdown_command_stops_worker(worker):
    reply = worker._process_command("goodbye")

    assert "shutting down" in reply.lower()
    assert worker._running is False


def test_chat_command_falls_back_to_offline_ai(worker, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "GROQ_AVAILABLE", False)
    monkeypatch.setattr(app_module, "GROQ_CLIENT", None)

    reply = worker._process_command("tell me a fun fact about space")

    assert "offline" in reply.lower()
