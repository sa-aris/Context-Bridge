"""The token-savings benchmark runs and shows sub-linear scaling."""

from __future__ import annotations

from context_bridge.benchmark import run_benchmark


def test_bridge_saves_tokens_at_scale():
    result = run_benchmark(num_agents=16, token_budget=128)

    assert result.baseline_tokens > 0
    assert result.bridge_tokens < result.baseline_tokens
    assert result.savings_pct > 0


def test_savings_grow_with_agent_count():
    small = run_benchmark(num_agents=8)
    large = run_benchmark(num_agents=24)

    # The shared-memory advantage widens as more agents join.
    assert large.savings_pct > small.savings_pct
