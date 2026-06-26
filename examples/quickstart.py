"""A self-contained tour of Context Bridge. Run it with::

    python examples/quickstart.py

It spins the memory manager up in-process (in-memory Qdrant + SQLite) with real
local embeddings via FastEmbed — a small model is downloaded once on first run —
then walks through the headline capabilities: shared writes from several agents,
token-budgeted semantic recall with provenance and a plain-language reason for
every hit, failure memory with a pre-task briefing, and the collaboration-quality
score. No external services required.
"""

from __future__ import annotations

import os
import time

from context_bridge.api.deps import build_container
from context_bridge.config import Settings

NS = "project-x"

# Optional cinematic pacing, used only when recording the demo animation.
# Real runs (CB_DEMO_DELAY unset) print instantly.
_DELAY = float(os.environ.get("CB_DEMO_DELAY", "0"))


def _pause(factor: float = 1.0) -> None:
    if _DELAY:
        time.sleep(_DELAY * factor)


def _rule(title: str) -> None:
    _pause(1.3)
    print(f"\n\033[1;36m── {title} \033[0m" + "─" * max(0, 58 - len(title)))
    _pause()


def main() -> None:
    settings = Settings(
        qdrant_url=":memory:",
        embed_provider="fastembed",  # real local embeddings (downloaded once)
        rerank_provider="identity",
        working_provider="memory",
        database_url="sqlite+pysqlite:///:memory:",
    )
    cb = build_container(settings).manager

    _rule("Three agents write into one shared pool")
    cb.write(
        content="The payment service uses Stripe and retries failed charges three times.",
        agent_id="billing-agent",
        session_id="run-42",
        namespace=NS,
    )
    cb.write(
        content="The office coffee machine is broken; a replacement is on order.",
        agent_id="ops-agent",
        session_id="run-42",
        namespace=NS,
    )
    cb.write(
        content="Checkout calls the payment service synchronously with a 5s timeout.",
        agent_id="api-agent",
        session_id="run-42",
        namespace=NS,
    )
    print("wrote 3 memories  ·  billing-agent, ops-agent, api-agent")

    _rule("A fourth agent recalls — only what the task needs")
    result = cb.query(
        query="how does the payment service handle failed charges?",
        namespace=NS,
        session_id="run-99",
        agent_id="reviewer",
        top_k=2,
        token_budget=256,
    )
    for chunk in result.chunks:
        print(f"  ✓ [{chunk.provenance.agent_id:>13}] {chunk.content}")
        print(f"      why: {_reason(chunk.signals)}")
        _pause()
    print(f"  tokens_used: {result.tokens_used} (budget 256)")
    print("  the coffee-machine note ranked last and was left out — recall stays relevant")

    _rule("Learn from a mistake, then brief the next agent before it starts")
    cb.record_lesson(
        namespace=NS,
        trigger="changing the payment service timeout",
        guidance="raise checkout's 5s timeout first, or synchronous calls will fail",
        severity="high",
    )
    brief = cb.preflight(task="tune the payment service timeout", namespace=NS)
    for lesson in brief["lessons"]:
        print(f"  [!] ({lesson['severity']}) {lesson['guidance']}")

    _rule("Close the loop, then read the namespace's pulse")
    cb.record_feedback(memory_id=result.chunks[0].id, namespace=NS, useful=True)
    q = cb.collaboration_quality(namespace=NS)
    print(f"  collaboration quality: {q['score']}/100")
    print(
        f"  hit_rate {q['hit_rate']}  ·  feedback_positivity {q['feedback_positivity']}"
        f"  ·  open_conflicts {q['open_conflicts']}"
    )

    print("\n\033[1;32m✓ shared, governed, explainable memory — all in-process\033[0m")


def _reason(signals: dict) -> str:
    bits = [f"match {signals.get('match')}"]
    if signals.get("age_days") is not None:
        bits.append("recent")
    return ", ".join(bits)


if __name__ == "__main__":
    main()
