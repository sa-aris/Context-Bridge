"""Entity-name normalization and canonical selection for ontology alignment."""

from __future__ import annotations

from context_bridge.core.graph.resolver import choose_canonical, normalize


def test_normalize_collapses_separators_and_case():
    assert normalize("Database-1") == "database 1"
    assert normalize("database  one") == "database one"
    assert normalize("  Service.Alpha  ") == "service alpha"
    assert normalize("cache_two") == "cache two"


def test_normalize_makes_variants_comparable():
    assert normalize("DB One") == normalize("db-one") == normalize("db_one")


def test_choose_canonical_prefers_most_connected():
    variants = ["db one", "Database One", "DB-1"]
    counts = {"Database One": 5, "db one": 2, "DB-1": 1}
    assert choose_canonical(variants, counts) == "Database One"


def test_choose_canonical_breaks_ties_by_length_then_alpha():
    variants = ["beta", "alpha", "gamma"]
    assert choose_canonical(variants, {}) == "beta"  # shortest (4); alpha/gamma are 5
    assert choose_canonical(["bbb", "aaa"], {}) == "aaa"  # equal length -> alphabetical
