"""HTTP surface for failure memory (lessons) and the preflight briefing."""

from __future__ import annotations


def test_lesson_record_and_list(client):
    resp = client.post(
        "/v1/lessons",
        json={
            "namespace": "ns",
            "trigger": "deploying to production on a friday",
            "guidance": "do not deploy on fridays",
            "severity": "high",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["id"]

    listed = client.get("/v1/lessons", params={"namespace": "ns"})
    assert listed.status_code == 200
    items = listed.json()["lessons"]
    assert items and items[0]["guidance"] == "do not deploy on fridays"


def test_query_returns_lessons_field(client):
    client.post(
        "/v1/lessons",
        json={
            "namespace": "qns",
            "trigger": "migrating the database schema",
            "guidance": "back up first",
        },
    )
    resp = client.post(
        "/v1/memory/query",
        json={"query": "migrating the database schema", "namespace": "qns"},
    )
    assert resp.status_code == 200
    assert "lessons" in resp.json()


def test_confirm_missing_lesson_404(client):
    resp = client.post("/v1/lessons/nope/confirm", params={"namespace": "ns"})
    assert resp.status_code == 404


def test_preflight_endpoint(client):
    client.post(
        "/v1/lessons",
        json={"namespace": "pns", "trigger": "scaling the cluster", "guidance": "watch quotas"},
    )
    resp = client.post("/v1/preflight", json={"task": "scaling the cluster", "namespace": "pns"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["task"] == "scaling the cluster"
    assert set(body) == {"task", "lessons", "procedures"}


def test_outcome_can_capture_lesson(client):
    client.post(
        "/v1/memory/write",
        json={"content": "did the thing", "agent_id": "a", "session_id": "run", "namespace": "ons"},
    )
    resp = client.post(
        "/v1/outcomes",
        json={
            "session_id": "run",
            "namespace": "ons",
            "success": False,
            "lesson": "check permissions before writing",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["lesson_id"]
