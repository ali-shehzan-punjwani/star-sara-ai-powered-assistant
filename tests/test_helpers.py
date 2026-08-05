"""Tests for the JSON persistence helpers and the fuzzy-similarity helper."""

import json

import pytest


# --------------------------------------------------------------------------
# save_json / load_json
# --------------------------------------------------------------------------

def test_load_json_creates_file_with_default_when_missing(app_module, tmp_path):
    path = tmp_path / "brand_new.json"
    default = {"hello": "world"}

    result = app_module.load_json(str(path), default)

    assert result == default
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == default


def test_save_then_load_round_trips_unicode(app_module, tmp_path):
    path = tmp_path / "data.json"
    payload = {"name": "STAR SARA", "emoji": "⭐", "nested": {"n": [1, 2, 3]}}

    app_module.save_json(str(path), payload)

    assert app_module.load_json(str(path), {}) == payload


def test_load_json_returns_default_on_corrupt_file(app_module, tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{ this is : not valid json", encoding="utf-8")

    result = app_module.load_json(str(path), {"fallback": True})

    assert result == {"fallback": True}


def test_save_json_swallows_errors(app_module, tmp_path, capsys):
    # A directory path can't be opened for writing as a file -> handled, no raise.
    directory = tmp_path / "a_directory"
    directory.mkdir()

    app_module.save_json(str(directory), {"x": 1})

    assert "[ERROR]" in capsys.readouterr().out


# --------------------------------------------------------------------------
# _text_similarity
# --------------------------------------------------------------------------

def test_text_similarity_identical_is_max(app_module):
    assert app_module._text_similarity("python", "python") == pytest.approx(100.0)


def test_text_similarity_is_case_and_whitespace_insensitive(app_module):
    assert app_module._text_similarity("Hello", "  hello  ") == pytest.approx(100.0)


@pytest.mark.parametrize("a,b", [("", "python"), ("python", ""), ("", "")])
def test_text_similarity_empty_operand_is_zero(app_module, a, b):
    assert app_module._text_similarity(a, b) == 0.0


def test_text_similarity_returns_float_in_range(app_module):
    score = app_module._text_similarity("banana", "orange")
    assert isinstance(score, float)
    assert 0.0 <= score <= 100.0


def test_text_similarity_more_alike_scores_higher(app_module):
    close = app_module._text_similarity("favorite language", "favourite language")
    far = app_module._text_similarity("favorite language", "zzzzzz qqqqqq")
    assert close > far
