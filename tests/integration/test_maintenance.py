"""TTL sweep: expired memories are physically removed, others survive."""

from __future__ import annotations

from context_bridge.core.memory.manager import MemoryManager


def test_sweep_removes_only_expired(manager: MemoryManager):
    expired = manager.write(
        content="This note self-destructs immediately.",
        agent_id="a",
        session_id="s",
        namespace="ns",
        ttl_seconds=0,
    )
    kept = manager.write(
        content="This note has no expiry and should remain.",
        agent_id="a",
        session_id="s",
        namespace="ns",
    )

    deleted = manager.sweep_expired()

    assert deleted == 1
    assert manager.get(expired.ids[0]) is None
    assert manager.get(kept.ids[0]) is not None


def test_sweep_endpoint(client):
    client.post(
        "/v1/memory/write",
        json={
            "content": "ephemeral api note",
            "agent_id": "a",
            "session_id": "s",
            "namespace": "ns",
            "ttl_seconds": 0,
        },
    )
    resp = client.post("/v1/maintenance/sweep")
    assert resp.status_code == 200
    assert resp.json()["deleted"] >= 1
