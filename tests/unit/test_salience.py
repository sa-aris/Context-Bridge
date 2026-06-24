from __future__ import annotations

from context_bridge.core.memory.salience import SalienceScorer


def test_emphasized_recurrent_turns_rank_highest():
    turns = [
        "hi there",
        "ok thanks",
        "IMPORTANT: we decided to migrate the database on Friday",
        "the database migration is the key decision for this project",
        "lol nice",
    ]
    top = SalienceScorer().distill(turns, max_promote=2)
    joined = " ".join(t.text.lower() for t in top)
    assert "migrat" in joined
    assert len(top) == 2


def test_chitchat_below_threshold_is_dropped():
    turns = ["hi", "ok", "yeah", "lol"]
    assert SalienceScorer(min_score=1.0).distill(turns) == []


def test_empty_turns():
    assert SalienceScorer().distill([]) == []


def test_max_promote_caps_results():
    turns = [f"the alpha beta gamma project milestone number {i} is important" for i in range(10)]
    assert len(SalienceScorer().distill(turns, max_promote=3)) == 3
