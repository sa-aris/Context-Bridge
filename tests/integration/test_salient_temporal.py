"""Cross-session salient memory and temporal (date-aware) recall."""

from __future__ import annotations

import time


def test_distilled_memory_survives_into_a_new_session(manager):
    # A noisy "chat": mostly chit-chat with a couple of dwelled-upon points.
    for text in [
        "hey",
        "ok",
        "IMPORTANT: the auth service must rotate tokens every 24 hours",
        "right, token rotation every 24 hours is the key rule for the auth service",
        "lol ok",
    ]:
        manager.remember_turn("chat-A", {"kind": "msg", "agent_id": "u", "content": text})

    out = manager.distill_session(session_id="chat-A", namespace="proj", max_promote=2)
    assert out["promoted"] >= 1

    # A brand-new session/chat in the same namespace recalls what mattered.
    result = manager.query(query="how often does the auth service rotate tokens?", namespace="proj")
    assert "rotate" in result.context.lower() or "rotation" in result.context.lower()


def test_include_dates_prefixes_context(manager):
    manager.write(
        content="the launch is scheduled for next quarter",
        agent_id="a",
        session_id="s",
        namespace="dt",
    )
    result = manager.query(query="launch schedule", namespace="dt", include_dates=True)
    assert result.context.startswith("[")  # "[YYYY-MM-DD] ..."


def test_temporal_since_filter(manager):
    manager.write(content="an old established fact", agent_id="a", session_id="s", namespace="tt")
    future = time.time() + 3600
    # Nothing is newer than 'future', so a since-filter past it returns empty.
    assert manager.query(query="fact", namespace="tt", since=future).chunks == []
    # Without the filter, it is recalled.
    assert manager.query(query="fact", namespace="tt").chunks


def test_session_turn_and_distill_endpoints(client):
    for text in ["hi", "we agreed the rate limit must be 100 rps for the gateway", "ok cool"]:
        r = client.post("/v1/sessions/chatX/turns", json={"agent_id": "u", "content": text})
        assert r.status_code == 204

    distilled = client.post(
        "/v1/sessions/chatX/distill", json={"namespace": "ops", "max_promote": 2}
    )
    assert distilled.status_code == 200
    assert distilled.json()["promoted"] >= 1

    recall = client.post(
        "/v1/memory/query",
        json={
            "query": "what is the gateway rate limit?",
            "namespace": "ops",
            "include_dates": True,
        },
    )
    assert recall.status_code == 200
    assert "100" in recall.json()["context"]
