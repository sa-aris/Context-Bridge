"""Role-based access control for namespaces.

Each API key resolves to a :class:`Rule`: a set of namespace glob patterns plus
the operations (``read`` / ``write``) it may perform. This is what makes the
shared pool safe for multi-tenant use — a key scoped to ``team-a*`` can never
read or write ``team-b`` memories.

Two configuration shapes feed it (rich overrides simple):

* ``API_KEY_NAMESPACES`` — ``{"key": ["ns-a", "ns-b"]}`` (implies read + write).
* ``API_KEY_POLICIES``   — ``{"key": {"namespaces": ["team-a*"],
  "permissions": ["read"]}}``.

With no API keys configured, access is open.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

READ = "read"
WRITE = "write"
_ALL = ("*",)
_RW = frozenset({READ, WRITE})


@dataclass(frozen=True)
class Rule:
    namespaces: tuple[str, ...]
    permissions: frozenset[str]


class AccessControl:
    """Resolves and enforces per-key namespace/operation rules."""

    def __init__(self, rules: dict[str, Rule], *, open_access: bool) -> None:
        self.rules = rules
        self.open_access = open_access

    @classmethod
    def build(cls, settings) -> AccessControl:
        keys = settings.api_key_set()
        rules: dict[str, Rule] = {}

        for key, namespaces in settings.api_key_namespace_map().items():
            rules[key] = Rule(tuple(namespaces) or _ALL, _RW)

        for key, policy in settings.api_key_policies_map().items():
            rules[key] = Rule(
                tuple(policy.get("namespaces") or _ALL),
                frozenset(policy.get("permissions") or [READ, WRITE]),
            )

        # Any configured key without an explicit rule gets full access.
        for key in keys:
            rules.setdefault(key, Rule(_ALL, _RW))

        return cls(rules, open_access=not keys)

    def allows(self, key: str | None, namespace: str, operation: str) -> bool:
        if self.open_access:
            return True
        rule = self.rules.get(key or "")
        if rule is None or operation not in rule.permissions:
            return False
        return any(fnmatch(namespace, pattern) for pattern in rule.namespaces)
