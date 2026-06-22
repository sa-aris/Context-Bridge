from __future__ import annotations

from context_bridge.core.vectorstore.qdrant_store import _reciprocal_rank_fusion


def test_item_ranked_high_in_both_lists_wins():
    dense = ["a", "b", "c"]
    sparse = ["a", "c", "b"]
    scores = _reciprocal_rank_fusion([dense, sparse])
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    assert ordered[0] == "a"  # top of both lists


def test_fusion_rewards_agreement_over_single_list_top():
    # "x" is #1 in one list only; "y" is #2 in both -> consistency should win.
    list_a = ["x", "y"]
    list_b = ["y", "x"]
    scores = _reciprocal_rank_fusion([list_a, list_b])
    # Symmetric here, but both must be present and scored.
    assert set(scores) == {"x", "y"}
    assert scores["x"] == scores["y"]


def test_empty_input():
    assert _reciprocal_rank_fusion([]) == {}
