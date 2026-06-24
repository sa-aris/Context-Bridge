from __future__ import annotations

from context_bridge.core.graph.extractor import RuleBasedExtractor
from context_bridge.core.memory.consolidation import cluster_by_similarity
from context_bridge.core.memory.contradiction import HeuristicDetector
from context_bridge.core.memory.redaction import NullRedactor, RegexRedactor


# -- redaction --------------------------------------------------------------
def test_regex_redactor_masks_pii():
    r = RegexRedactor()
    out = r.redact("email me at jane.doe@example.com or call 555-123-4567")
    assert "jane.doe@example.com" not in out
    assert "[REDACTED:EMAIL]" in out


def test_null_redactor_is_passthrough():
    assert (
        NullRedactor().redact("secret sk-abcdefghijklmnop1234") == "secret sk-abcdefghijklmnop1234"
    )


# -- contradiction ----------------------------------------------------------
def test_negation_flip_is_contradiction():
    d = HeuristicDetector()
    assert d.is_contradiction(
        "the deployment is enabled for the api gateway",
        "the deployment is not enabled for the api gateway",
    )


def test_number_mismatch_is_contradiction():
    d = HeuristicDetector()
    assert d.is_contradiction(
        "the retry budget for the worker is 3 attempts",
        "the retry budget for the worker is 5 attempts",
    )


def test_unrelated_is_not_contradiction():
    d = HeuristicDetector()
    assert not d.is_contradiction("the cat sat on the mat", "kubernetes schedules pods on nodes")


# -- consolidation clustering ----------------------------------------------
def test_cluster_groups_similar_vectors():
    vecs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    clusters = cluster_by_similarity(vecs, threshold=0.9)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_cluster_empty():
    assert cluster_by_similarity([], threshold=0.9) == []


# -- graph extraction -------------------------------------------------------
def test_rule_based_extractor_finds_triples():
    triples = RuleBasedExtractor().extract("service alpha depends on database one.")
    assert ("service alpha", "depends on", "database one") in [
        (t.source, t.relation, t.target) for t in triples
    ]


def test_extractor_ignores_unrelated_text():
    assert RuleBasedExtractor().extract("hello there, nice weather today") == []
