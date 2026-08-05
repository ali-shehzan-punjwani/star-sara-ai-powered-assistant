"""Tests for AIEngine task management, note management, profile safety, and Groq wrapper."""

from datetime import datetime, timedelta


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def test_add_task_stores_pending_task(ai):
    ai.add_task("finish report", priority="high", due="2030-01-01")

    tasks = ai.tasks["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["task"] == "finish report"
    assert tasks[0]["priority"] == "high"
    assert tasks[0]["status"] == "pending"
    assert tasks[0]["due"] == "2030-01-01"


def test_get_pending_tasks_excludes_completed(ai):
    ai.add_task("task one")
    ai.add_task("task two")
    ai.complete_task(0)

    pending = ai.get_pending_tasks()

    assert len(pending) == 1
    assert pending[0]["task"] == "task two"


def test_complete_task_valid_index(ai):
    ai.add_task("do it")
    assert ai.complete_task(0) is True
    assert ai.tasks["tasks"][0]["status"] == "completed"


def test_complete_task_invalid_index_returns_false(ai):
    ai.add_task("only one")
    assert ai.complete_task(99) is False


def test_get_tasks_due_today(ai):
    today = datetime.now().date().isoformat()
    tomorrow = (datetime.now().date() + timedelta(days=1)).isoformat()
    ai.add_task("due now", due=today)
    ai.add_task("later", due=tomorrow)

    due = ai.get_tasks_due_today()

    assert len(due) == 1
    assert due[0]["task"] == "due now"


def test_format_tasks_empty(ai):
    assert "no pending tasks" in ai.format_tasks().lower()


def test_format_tasks_lists_tasks_with_due(ai):
    ai.add_task("call the bank", due="2030-05-05")

    text = ai.format_tasks()

    assert "1 pending tasks" in text
    assert "call the bank" in text
    assert "due 2030-05-05" in text


def test_format_due_today_empty_and_populated(ai):
    assert "nothing is due today" in ai.format_due_today().lower()

    today = datetime.now().date().isoformat()
    ai.add_task("submit form", due=today)
    text = ai.format_due_today()
    assert "due today" in text.lower()
    assert "submit form" in text


# --------------------------------------------------------------------------
# notes
# --------------------------------------------------------------------------

def test_save_note_and_format(ai):
    ai.save_note("Groceries", "buy milk and eggs")

    text = ai.format_notes()

    assert "1 saved notes" in text
    assert "Groceries" in text
    assert "buy milk and eggs" in text


def test_format_notes_empty(ai):
    assert "do not have any saved notes" in ai.format_notes().lower()


def test_search_notes_empty_store(ai):
    assert "do not have any saved notes" in ai.search_notes("anything").lower()


def test_search_notes_finds_match(ai):
    ai.save_note("Groceries", "buy milk and eggs")

    result = ai.search_notes("groceries buy milk")

    assert "Groceries" in result
    assert "buy milk and eggs" in result


def test_search_notes_no_match(ai):
    ai.save_note("Groceries", "buy milk and eggs")

    result = ai.search_notes("zzz totally unrelated qqq")

    assert "couldn't find" in result.lower()


# --------------------------------------------------------------------------
# _llm_safe_profile
# --------------------------------------------------------------------------

def test_llm_safe_profile_strips_sensitive_sections_and_keys(ai):
    ai.user_data = {
        "identity": {
            "preferred_name": "Shehzan",
            "city": "Karachi",
            "date_of_birth": "1900-01-01",
            "religion": "private",
            "age": 25,
            "gender": "private",
        },
        "contact": {"email": "secret@example.com"},
        "family": {"mother": "private"},
        "career": {"target_role": "CISO"},
    }

    safe = ai._llm_safe_profile()

    assert "contact" not in safe
    assert "family" not in safe
    assert safe["career"] == {"target_role": "CISO"}
    assert safe["identity"] == {"preferred_name": "Shehzan", "city": "Karachi"}


# --------------------------------------------------------------------------
# build_context / add_turn / ask
# --------------------------------------------------------------------------

def test_build_context_mentions_owner_and_task_counts(ai, app_module):
    ai.add_task("something")
    context = ai.build_context("hello there")

    assert app_module.ASSISTANT_NAME in context
    assert app_module.OWNER_ADDRESS in context
    assert "1 pending" in context


def test_build_context_no_memories(ai):
    assert "No relevant memories stored." in ai.build_context("random query")


def test_add_turn_is_bounded(ai, app_module):
    limit = app_module.CONVERSATION_HISTORY_TURNS * 2
    for i in range(limit + 10):
        ai.add_turn("user", f"message {i}")

    assert len(ai.conversation_history) == limit


def test_ask_returns_offline_message_when_groq_unavailable(ai, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "GROQ_AVAILABLE", False)
    monkeypatch.setattr(app_module, "GROQ_CLIENT", None)

    reply = ai.ask("what is the weather")

    assert "offline" in reply.lower()
    # a failed/offline call must not pollute conversation history
    assert len(ai.conversation_history) == 0
