# Context Bridge

**Shared neural memory middleware for multi-agent systems.**

Multi-agent frameworks route work by passing ever-larger context blobs between
agents. Token cost grows quadratically and the relevant signal drowns in noise.
Context Bridge replaces that with a **shared memory pool**: agents *write* their
outputs into a common store and *recall* only the task-specific, token-budgeted
slice they need — via hybrid retrieval, reranking and diversification.

## Why it is different

- **Shared, not per-agent.** Memory is a first-class service every agent reads
  from and writes to, with full provenance — not a private buffer.
- **Governed writes.** Confidence gating + near-duplicate suppression +
  optional summarise-before-store keep the pool from poisoning itself.
- **Token budgets are explicit.** Every recall is bounded by a token budget and
  reports exactly what it spent — the measurable cost-savings lever.

## Architecture: three memory tiers

| Tier | Purpose | Backing store |
| --- | --- | --- |
| **Working** | Recent per-session scratchpad, ephemeral | In-process / Redis |
| **Semantic** | Long-term embedded knowledge | Qdrant (dense + sparse) |
| **Episodic** | Task graph & provenance ("who/what/when/why") | SQL (SQLite / Postgres) |

**Write path:** `chunk → embed (dense + sparse) → govern (dedup / confidence) →
persist → log episode`.

**Read path:** `embed query → hybrid search (RRF fusion) → drop expired →
cross-encoder rerank → MMR diversify → assemble within token budget`.

Every provider sits behind a small protocol, so the vector store
(`VectorStore`), embedder (`Embedder`), reranker (`Reranker`) and working store
(`WorkingMemory`) are swappable without touching the rest of the system.

## Quick start

```bash
# Optional backing services (Qdrant + Postgres + Redis)
docker compose up -d

# Install (uv recommended)
uv pip install -e ".[dev]"          # core + tooling
# uv pip install -e ".[dev,fastembed,redis,postgres]"   # full local stack

# Run the API
make run          # http://localhost:8000/docs
```

Out of the box the defaults are dependency-light and fully offline: an
in-process Qdrant (`QDRANT_URL=:memory:`), a deterministic hashing embedder and
a SQLite database. Switch `EMBED_PROVIDER=fastembed` for production-quality
local embeddings and point `QDRANT_URL` / `DATABASE_URL` at real services. See
`.env.example` for all options.

## Using it from an agent

```python
from context_bridge.sdk import ContextBridgeClient

with ContextBridgeClient("http://localhost:8000") as cb:
    cb.remember(
        "The payment service retries failed charges three times.",
        agent_id="billing-agent",
        session_id="run-42",
        namespace="project-x",
    )

    result = cb.recall(
        "how are failed charges handled?",
        namespace="project-x",
        token_budget=512,
    )
    print(result["context"])   # budget-bounded, reranked, deduped
    print(result["sources"])   # provenance for every included chunk
```

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/memory/write` | Chunk, embed, govern and store content |
| `POST` | `/memory/query` | Hybrid recall within a token budget |
| `GET` | `/memory/{id}` | Fetch a single stored record |
| `DELETE` | `/memory/{id}` | Remove a record |
| `POST` | `/memory/summarize` | Compress a session into a summary memory |
| `GET` | `/sessions/{id}/timeline` | Episodic / provenance view |
| `GET` | `/health`, `/healthz` | Liveness / readiness |

## Development

```bash
make test        # pytest (hermetic: in-memory Qdrant + SQLite)
make lint        # ruff
make typecheck   # mypy
```

Database migrations are managed with Alembic (`alembic upgrade head`); for local
dev the API auto-creates tables on startup.

## License

MIT
