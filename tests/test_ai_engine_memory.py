"""Tests for AIEngine's memory subsystem (remember/recall/rank/decay/migrate)."""

from datetime import datetime, timedelta


# --------------------------------------------------------------------------
# remember
# --------------------------------------------------------------------------

def test_remember_appends_new_fact(ai, app_module):
    msg = ai.remember("lang", "my favorite language is python")

    assert app_module.OWNER_ADDRESS in msg
    facts = ai.memory["facts"]
    assert len(facts) == 1
    fact = facts[0]
    assert fact["key"] == "lang"
    assert fact["value"] == "my favorite language is python"
    assert fact["importance"] == 3
    assert fact["access_count"] == 0
    assert "created_at" in fact and "last_accessed" in fact


def test_remember_deduplicates_near_identical_value(ai):
    ai.remember("a", "my favorite language is python")
    msg = ai.remember("b", "my favorite language is python")

    assert "updated" in msg.lower()
    assert len(ai.memory["facts"]) == 1


def test_remember_keeps_highest_importance_on_update(ai):
    ai.remember("a", "critical fact about deadlines", importance=5)
    ai.remember("b", "critical fact about deadlines", importance=2)

    assert ai.memory["facts"][0]["importance"] == 5


def test_remember_enforces_max_facts_bound(ai, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MEMORY_MAX_FACTS", 2)

    ai.remember("k1", "alpha distinct fact one", importance=5)
    ai.remember("k2", "beta distinct fact two", importance=5)
    ai.remember("k3", "gamma distinct fact three", importance=1)

    facts = ai.memory["facts"]
    assert len(facts) == 2
    # the lowest-importance fact should have been evicted
    values = {f["value"] for f in facts}
    assert "gamma distinct fact three" not in values


# --------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------

def test_recall_returns_value_and_bumps_access(ai):
    ai.remember("lang", "python is my favorite")
    before = ai.memory["facts"][0]["access_count"]

    value = ai.recall("lang")

    assert value == "python is my favorite"
    assert ai.memory["facts"][0]["access_count"] == before + 1


def test_recall_missing_key_returns_none(ai):
    assert ai.recall("does-not-exist") is None


# --------------------------------------------------------------------------
# relevant_memories
# --------------------------------------------------------------------------

def test_relevant_memories_empty_store(ai):
    assert ai.relevant_memories("anything") == []


def test_relevant_memories_returns_matching_fact(ai):
    ai.remember("lang", "my favorite programming language is python")

    results = ai.relevant_memories("what programming language do i like")

    assert len(results) == 1
    assert results[0]["value"] == "my favorite programming language is python"
    # touching a fact bumps its access stats
    assert results[0]["access_count"] >= 1


def test_relevant_memories_filters_low_score(ai):
    ai.remember("k", "purple monkey dishwasher", importance=1)

    # A query with no character overlap scores below the relevance floor.
    assert ai.relevant_memories("wxyz wxyz wxyz") == []


def test_relevant_memories_respects_top_k(ai):
    ai.remember("k1", "python programming language notes")
    ai.remember("k2", "python programming tips and tricks")
    ai.remember("k3", "python programming best practices")

    results = ai.relevant_memories("python programming", top_k=1)

    assert len(results) == 1


# --------------------------------------------------------------------------
# _decay_memory
# --------------------------------------------------------------------------

def test_decay_drops_stale_low_importance_facts(ai, app_module):
    old = (datetime.now() - timedelta(days=app_module.MEMORY_DECAY_DAYS + 5)).isoformat()
    ai.memory["facts"] = [
        {"key": "stale", "value": "old trivia", "importance": 1,
         "created_at": old, "last_accessed": old, "access_count": 0},
        {"key": "keep", "value": "important thing", "importance": 5,
         "created_at": old, "last_accessed": old, "access_count": 0},
    ]

    ai._decay_memory()

    remaining = {f["key"] for f in ai.memory["facts"]}
    assert remaining == {"keep"}


def test_decay_keeps_recent_facts(ai):
    now = datetime.now().isoformat()
    ai.memory["facts"] = [
        {"key": "recent", "value": "fresh fact", "importance": 1,
         "created_at": now, "last_accessed": now, "access_count": 0},
    ]

    ai._decay_memory()

    assert len(ai.memory["facts"]) == 1


# --------------------------------------------------------------------------
# _migrate_memory_schema
# --------------------------------------------------------------------------

def test_migrate_upgrades_legacy_facts(ai):
    ai.memory = {"facts": [{"key": "k", "value": "v"}]}

    ai._migrate_memory_schema()

    fact = ai.memory["facts"][0]
    assert fact["importance"] == 3
    assert fact["access_count"] == 0
    assert "created_at" in fact
    assert fact["last_accessed"] == fact["created_at"]
