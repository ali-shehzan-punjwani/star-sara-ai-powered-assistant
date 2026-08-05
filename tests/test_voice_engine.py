"""Tests for the pure (non-hardware) VoiceEngine methods."""

import pytest


@pytest.fixture
def voice(app_module):
    return app_module.VoiceEngine()


# --------------------------------------------------------------------------
# remove_wake_word (static)
# --------------------------------------------------------------------------

def test_remove_wake_word_strips_wake_phrase(app_module):
    assert app_module.VoiceEngine.remove_wake_word("star sara add task") == "add task"


def test_remove_wake_word_case_insensitive(app_module):
    assert app_module.VoiceEngine.remove_wake_word("STAR SARA what time is it") == "what time is it"


# --------------------------------------------------------------------------
# _is_likely_hallucination (static)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("hello hello hello", True),        # 1 unique word repeated
    ("hello hi hello hi", True),        # 2 unique words repeated
    ("what is the weather today", False),
    ("hi hi", False),                   # fewer than 3 tokens -> not judged
])
def test_is_likely_hallucination(app_module, text, expected):
    assert app_module.VoiceEngine._is_likely_hallucination(text) is expected


# --------------------------------------------------------------------------
# contains_wake_word
# --------------------------------------------------------------------------

def test_contains_wake_word_none_is_false(voice):
    assert voice.contains_wake_word(None) is False


def test_contains_wake_word_exact_match(voice):
    assert voice.contains_wake_word("star sara are you there") is True


def test_contains_wake_word_word_boundary_avoids_false_positive(voice):
    # "sarah" must not match the "sara" wake word.
    assert voice.contains_wake_word("my friend sarah is here") is False


def test_contains_wake_word_no_wake_word(voice):
    assert voice.contains_wake_word("what time is it") is False


def test_contains_wake_word_cooldown_suppresses_repeat(voice):
    assert voice.contains_wake_word("hey star sara") is True
    # An immediate second trigger is swallowed by the cooldown window.
    assert voice.contains_wake_word("star sara again") is False
