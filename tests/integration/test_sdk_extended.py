"""The synchronous SDK client against the in-process app (extended surface)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from context_bridge.api.app import create_app
from context_bridge.sdk.client import ContextBridgeClient


def test_sdk_learning_and_insight_surface(settings):
    app = create_app(settings)
    # TestClient runs the app lifespan (building the manager) and the SDK drives
    # it directly by pointing its transport at the in-process app under /v1.
    with TestClient(app, base_url="http://testserver/v1") as test_client:
        sdk = ContextBridgeClient()
        sdk._client = test_client
        try:
            written = sdk.remember(
                "the deploy uses blue-green cutover",
                agent_id="a",
                session_id="s",
                namespace="sdk",
            )
            memory_id = written["ids"][0]

            # learning loop
            sdk.feedback(memory_id, namespace="sdk", useful=True, weight=2.0)
            sdk.record_outcome("s", namespace="sdk", success=True)

            # failure memory + preflight
            sdk.record_lesson("blue-green cutover", "drain connections first", namespace="sdk")
            assert sdk.lessons(namespace="sdk")["lessons"]
            brief = sdk.preflight("blue-green cutover", namespace="sdk")
            assert "lessons" in brief

            # insight surface
            assert "score" in sdk.quality(namespace="sdk")
            health = sdk.health(namespace="sdk")
            assert health["namespace"] == "sdk"
            assert "events" in sdk.beliefs("deploy cutover", namespace="sdk")

            # operations: export -> import round-trip
            dump = sdk.export_namespace(namespace="sdk")
            counts = sdk.import_namespace(dump, namespace="sdk_copy")
            assert counts["memories"] >= 1

            assert set(sdk.run_maintenance()) == {
                "swept",
                "namespaces",
                "conflicts_resolved",
                "insights",
                "lessons_created",
            }
        finally:
            sdk.close()
