"""HTTP surface for maintenance run and namespace import/export."""

from __future__ import annotations


def _write(client, content, namespace):
    return client.post(
        "/v1/memory/write",
        json={"content": content, "agent_id": "a", "session_id": "s", "namespace": namespace},
    )


def test_maintenance_run_endpoint(client):
    resp = client.post("/v1/maintenance/run")
    assert resp.status_code == 200
    assert set(resp.json()) == {
        "swept",
        "namespaces",
        "conflicts_resolved",
        "insights",
        "lessons_created",
    }


def test_export_import_endpoints(client):
    _write(client, "the worker scales to ten replicas", "src")
    client.post(
        "/v1/lessons",
        json={"namespace": "src", "trigger": "scaling workers", "guidance": "watch memory"},
    )

    export = client.get("/v1/namespaces/src/export")
    assert export.status_code == 200
    dump = export.json()
    assert dump["namespace"] == "src"
    assert dump["memories"]

    imp = client.post("/v1/namespaces/dst/import", json=dump)
    assert imp.status_code == 200
    body = imp.json()
    assert body["memories"] >= 1
    assert body["lessons"] >= 1
