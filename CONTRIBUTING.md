# Contributing to Context Bridge

Thanks for your interest in improving Context Bridge! This guide covers the
basics for getting set up and submitting changes.

## Development setup

```bash
# Create an environment and install with dev extras
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Optional: spin up Qdrant + Postgres + Redis locally
docker compose up -d
```

## Before you open a pull request

Run the full local gate — CI runs the same checks:

```bash
ruff check src tests        # lint
ruff format --check src tests
mypy src                    # type-check
pytest                      # tests (hermetic; no network needed)
```

The default test suite is fully offline: it uses an in-process Qdrant
(`:memory:`), SQLite, and a deterministic hashing embedder. Tests that need a
downloaded model (e.g. the FastEmbed reranker smoke test) skip automatically
when the model is unavailable.

## Guidelines

- **Keep providers behind their protocols.** New vector stores, embedders,
  rerankers, working-memory or summarizer backends should implement the
  existing `Protocol` in the relevant `core/*` package — don't couple callers
  to a concrete backend.
- **Add tests** for new behavior. Prefer hermetic tests; gate anything needing
  network/models behind `pytest.importorskip` / `pytest.skip`.
- **Conventional commits.** Use prefixes like `feat:`, `fix:`, `docs:`,
  `test:`, `chore:`, `refactor:`.
- **Type everything.** `mypy src` must pass.

## Cutting a release

Releases are tag-driven. Bump `version` in `pyproject.toml`, update
`CHANGELOG.md`, then:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Release` workflow builds the sdist/wheel and creates a GitHub Release with
generated notes. To also publish to PyPI, do the one-time setup:

1. On PyPI, add a **trusted publisher** for this repo (workflow `release.yml`,
   environment `pypi`) — no token needed.
2. Create a GitHub Actions **environment** named `pypi`.
3. Set the repository **variable** `PYPI_ENABLED=true`.

Until then the PyPI job is skipped, so tagging still cuts a clean GitHub Release.

## Reporting bugs / requesting features

Open an issue using the provided templates. For security issues, please follow
[SECURITY.md](SECURITY.md) instead of filing a public issue.
