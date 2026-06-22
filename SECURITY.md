# Security Policy

## Supported versions

Context Bridge is pre-1.0; security fixes are applied to the `main` branch.

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Instead, report privately via GitHub's
[security advisories](https://github.com/sa-aris/context-bridge/security/advisories/new)
("Report a vulnerability"). Include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- any suggested remediation.

You can expect an initial acknowledgement within a few business days. We will
coordinate a fix and disclosure timeline with you.

## Hardening notes

- Set `API_KEYS` to require authentication and scope keys to namespaces via
  `API_KEY_NAMESPACES` for multi-tenant deployments.
- Put the service behind TLS and a reverse proxy in production.
- Use the Redis rate-limiter backend (`RATE_LIMIT_BACKEND=redis`) when running
  more than one replica.
- Treat the vector store, database and Redis as trusted internal services and
  isolate them at the network level.
