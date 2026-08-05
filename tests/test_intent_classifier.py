"""Tests for the rule-based IntentClassifier."""

import pytest


@pytest.fixture
def classify(app_module):
    return app_module.IntentClassifier.classify


@pytest.mark.parametrize("command,expected", [
    ("add task buy milk", "add_task"),
    ("new task call the dentist", "add_task"),
    ("remember that I like tea", "remember"),
    ("what are my tasks", "list_tasks"),
    ("show me pending tasks", "list_tasks"),
    ("what's due today", "tasks_due_today"),
    ("take a note buy bread", "save_note"),
    ("read my notes", "list_notes"),
    ("find my note about cars", "search_notes"),
    ("goodbye", "shutdown"),
    ("please stop", "shutdown"),
])
def test_exact_keyword_intents(classify, command, expected):
    assert classify(command) == expected


def test_classify_is_case_insensitive(classify):
    assert classify("ADD TASK buy milk") == "add_task"


def test_free_form_chat_defaults_to_chat(classify):
    assert classify("what is the capital of france") == "chat"


def test_word_boundary_prevents_false_shutdown(classify):
    # "stopping" contains "stop" but must NOT trigger the shutdown intent.
    assert classify("i am stopping by the store later") != "shutdown"


def test_long_unrelated_sentence_is_chat(classify):
    command = "could you help me understand how neural networks actually learn"
    assert classify(command) == "chat"
