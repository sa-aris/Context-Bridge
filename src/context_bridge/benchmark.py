"""Quantify the token savings of shared memory vs. transcript passing.

Two scenarios run over the same synthetic multi-agent task:

* **Baseline** — every agent is handed the full, growing transcript of all
  prior agent outputs (the AutoGen-style default). Context cost grows roughly
  quadratically with the number of agents.
* **Context Bridge** — every agent writes its output to shared memory and
  recalls only a token-budgeted slice relevant to its step.

Run it directly to print a comparison:

    python -m context_bridge.benchmark
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

from context_bridge.api.deps import build_container
from context_bridge.config import Settings
from context_bridge.tokenizer import count_tokens


@dataclass(slots=True)
class BenchmarkResult:
    num_agents: int
    token_budget: int
    baseline_tokens: int
    bridge_tokens: int

    @property
    def savings_pct(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return 100.0 * (self.baseline_tokens - self.bridge_tokens) / self.baseline_tokens


def _synthetic_outputs(num_agents: int) -> list[str]:
    """Distinct, lexically-searchable outputs, one per agent step."""
    return [
        (
            f"Step {i}: component service_{i} was configured. "
            f"Its retry budget is {3 + i} attempts and it depends on database_{i} "
            f"for persistence, exposing endpoint_{i} to downstream callers."
        )
        for i in range(num_agents)
    ]


def _query_for(step: int) -> str:
    """Each agent asks about the component produced one step earlier."""
    return f"what is the retry budget and dependency of service_{step - 1}?"


def run_benchmark(*, num_agents: int = 8, token_budget: int = 128) -> BenchmarkResult:
    """Run both scenarios on an in-process stack and return the token totals."""
    outputs = _synthetic_outputs(num_agents)

    # --- Baseline: each agent reads the full transcript so far ---
    baseline_tokens = 0
    transcript: list[str] = []
    for output in outputs:
        baseline_tokens += count_tokens("\n".join(transcript))
        transcript.append(output)

    # --- Context Bridge: each agent recalls only a budgeted slice ---
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(
            qdrant_url=":memory:",
            qdrant_collection="benchmark",
            embed_provider="hashing",
            embed_dim=128,
            rerank_provider="identity",
            working_provider="memory",
            database_url=f"sqlite+pysqlite:///{Path(tmp) / 'bench.db'}",
        )
        manager = build_container(settings).manager

        bridge_tokens = 0
        for step, output in enumerate(outputs):
            if step > 0:
                recalled = manager.query(
                    query=_query_for(step),
                    namespace="bench",
                    token_budget=token_budget,
                )
                bridge_tokens += recalled.tokens_used
            manager.write(
                content=output,
                agent_id=f"agent-{step}",
                session_id="bench",
                namespace="bench",
            )

    return BenchmarkResult(
        num_agents=num_agents,
        token_budget=token_budget,
        baseline_tokens=baseline_tokens,
        bridge_tokens=bridge_tokens,
    )


def main() -> None:
    print(f"{'agents':>7} {'transcript':>12} {'bridge':>10} {'savings':>9}")
    print("-" * 42)
    for num_agents in (4, 8, 16, 32):
        result = run_benchmark(num_agents=num_agents)
        print(
            f"{result.num_agents:>7} "
            f"{result.baseline_tokens:>12,} "
            f"{result.bridge_tokens:>10,} "
            f"{result.savings_pct:>8.1f}%"
        )


if __name__ == "__main__":
    main()
