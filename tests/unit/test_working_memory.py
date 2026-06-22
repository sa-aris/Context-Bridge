from __future__ import annotations

from context_bridge.core.working.memory_store import InMemoryWorkingStore


def test_append_and_recent_oldest_first():
    store = InMemoryWorkingStore(ttl_seconds=3600)
    for i in range(3):
        store.append("s", {"i": i})

    recent = store.recent("s", limit=10)
    assert [item["i"] for item in recent] == [0, 1, 2]


def test_recent_respects_limit():
    store = InMemoryWorkingStore(ttl_seconds=3600)
    for i in range(10):
        store.append("s", {"i": i})

    recent = store.recent("s", limit=3)
    assert [item["i"] for item in recent] == [7, 8, 9]


def test_ttl_expires_entries():
    store = InMemoryWorkingStore(ttl_seconds=0)
    store.append("s", {"i": 1})
    assert store.recent("s") == []


def test_clear_removes_session():
    store = InMemoryWorkingStore(ttl_seconds=3600)
    store.append("s", {"i": 1})
    store.clear("s")
    assert store.recent("s") == []


def test_sessions_are_isolated():
    store = InMemoryWorkingStore(ttl_seconds=3600)
    store.append("a", {"v": "x"})
    store.append("b", {"v": "y"})
    assert store.recent("a")[0]["v"] == "x"
    assert store.recent("b")[0]["v"] == "y"
