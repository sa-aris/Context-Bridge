<div align="center">

# 🧠 Context Bridge

### Shared neural memory middleware for multi-agent systems

*Stop passing giant transcripts between agents. Give them a shared memory and let each one recall only what it needs.*

[![CI](https://github.com/sa-aris/context-bridge/actions/workflows/ci.yml/badge.svg)](https://github.com/sa-aris/context-bridge/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Typed](https://img.shields.io/badge/typed-mypy-blue.svg)](http://mypy-lang.org/)

</div>

---

## The problem

Multi-agent frameworks route work by **passing ever-larger context blobs** from
agent to agent. The shared history grows with every hop, so token cost climbs
super-linearly and the genuinely relevant facts drown in noise.

```text
        ❌ Without Context Bridge                  ✅ With Context Bridge

  Agent A ──[full history]──▶ Agent B          Agent A ─┐        ┌─ Agent B
     │                          │                       ▼        ▼
     └──[bigger history]──▶ Agent C             ┌─────────────────────┐
            tokens 📈📈📈 cost 💸               │   Shared memory pool │
                                                │  (write • recall)    │
   every agent re-reads everything              └─────────────────────┘
                                                each agent recalls only its
                                                task-scoped, budgeted slice
```

Context Bridge is a standalone service that turns agent memory into a **shared,
governed, queryable pool**. Agents *write* their outputs into it and *recall*
just the slice they need for the current task — under a strict token budget.

## Why it's different

- **Shared, not per-agent.** Memory is a first-class service every agent reads
  from and writes to — with full provenance — not a private scratch buffer.
- **Governed writes.** Confidence gating, near-duplicate suppression and
  optional summarize-before-store stop the pool from poisoning itself.
- **Token budgets are explicit.** Every recall is bounded and reports exactly
  what it spent — the measurable cost-savings lever.

## Architecture

```mermaid
flowchart TB
    subgraph Agents
      A1[Agent A]
      A2[Agent B]
      A3[Agent C]
    end

    subgraph CB["Context Bridge API (/v1)"]
      MM[MemoryManager]
    end

    subgraph Tiers["Memory tiers"]
      W[("Working memory<br/>in-process / Redis")]
      S[("Semantic memory<br/>Qdrant · dense + sparse")]
      E[("Episodic / provenance<br/>SQL · SQLAlchemy")]
    end

    A1 & A2 & A3 -->|remember / recall| MM
    MM --> W
    MM --> S
    MM --> E
```

| Tier | Purpose | Backing store |
| --- | --- | --- |
| **Working** | Recent per-session scratchpad, ephemeral | In-process / Redis |
| **Semantic** | Long-term embedded knowledge | Qdrant (dense + sparse) |
| **Episodic** | Task graph & provenance ("who/what/when/why") | SQL (SQLite / Postgres) |

### Read & write pipelines

```mermaid
sequenceDiagram
    participant Agent
    participant API as Context Bridge
    participant V as Qdrant
    participant DB as SQL

    Note over Agent,DB: WRITE (governed)
    Agent->>API: remember(content, agent_id, namespace)
    API->>API: chunk → embed (dense+sparse)
    API->>API: dedup + confidence gate
    API->>V: upsert chunks
    API->>DB: store parent doc + episode

    Note over Agent,DB: RECALL (token-budgeted)
    Agent->>API: recall(query, token_budget)
    API->>V: hybrid search (RRF fusion)
    API->>API: rerank → MMR → assemble ≤ budget
    API-->>Agent: context + cited sources + tokens_used
```

Every provider sits behind a small `Protocol` — `VectorStore`, `Embedder`,
`Reranker`, `WorkingMemory`, `Summarizer` — so backends are swappable without
vendor lock-in.

## Features

| | |
| --- | --- |
| 🔀 **Hybrid retrieval** | Dense + sparse vectors fused with Reciprocal Rank Fusion |
| 🎯 **Reranking + MMR** | Cross-encoder precision, then diversity to kill redundancy |
| 💰 **Token budgeting** | Recall is bounded and reports `tokens_used` |
| 🧬 **Small-to-big chunks** | Match a small chunk, expand to its parent on demand |
| 🛡️ **Write governance** | Dedup, confidence gating, summarize-before-store |
| 🧾 **Provenance** | Every memory is attributable; full session timeline |
| ⏳ **TTL & decay** | Expiry at query time + background/endpoint sweep |
| 🏢 **Multi-tenant** | API keys scoped to namespaces |
| 🔑 **Auth & rate limit** | Constant-time keys; in-memory or Redis limiter |
| 📊 **Observability** | Prometheus metrics, request IDs, structured errors |
| 🐳 **Production-ready** | Dockerfile, CI, Alembic migrations, typed SDK |
| 🔌 **Pluggable** | Swap Qdrant/embedder/reranker behind protocols |

## Quick start

```bash
# 1. (optional) backing services — Qdrant + Postgres + Redis
docker compose up -d

# 2. install
uv pip install -e ".[dev]"            # core + tooling
# uv pip install -e ".[fastembed,redis,postgres]"   # full local stack

# 3. run
make run                              # ->  http://localhost:8000/docs
```

Out of the box the defaults are **dependency-light and fully offline**: an
in-process Qdrant (`QDRANT_URL=:memory:`), a deterministic hashing embedder and
SQLite. Flip `EMBED_PROVIDER=fastembed` for production-quality local embeddings
and point `QDRANT_URL` / `DATABASE_URL` at real services. See
[`.env.example`](.env.example) for every option.

## Demo

**Write two memories from two different agents, then recall just the relevant one:**

```bash
# Agent 1 writes
curl -s localhost:8000/v1/memory/write -H 'content-type: application/json' -d '{
  "content": "The payment service uses Stripe and retries failed charges three times.",
  "agent_id": "billing-agent", "session_id": "run-42", "namespace": "project-x"
}'

# Agent 2 writes something unrelated
curl -s localhost:8000/v1/memory/write -H 'content-type: application/json' -d '{
  "content": "The office coffee machine is broken; a replacement is on order.",
  "agent_id": "ops-agent", "session_id": "run-42", "namespace": "project-x"
}'

# Another agent recalls — within a token budget
curl -s localhost:8000/v1/memory/query -H 'content-type: application/json' -d '{
  "query": "how does the payment service handle failed charges?",
  "namespace": "project-x", "token_budget": 256
}'
```

```jsonc
{
  "context": "The payment service uses Stripe and retries failed charges three times.",
  "tokens_used": 14,
  "chunks": [{ "id": "…", "agent_id": "billing-agent", "score": 0.83, … }],
  "sources": [{ "agent_id": "billing-agent", "session_id": "run-42", … }]
}
```

The coffee-machine note never shows up — only the budgeted, reranked, deduped
slice the query actually needs, with provenance for every line.

### From an agent (Python SDK)

```python
from context_bridge.sdk import ContextBridgeClient

with ContextBridgeClient("http://localhost:8000") as cb:
    cb.remember(
        "The payment service retries failed charges three times.",
        agent_id="billing-agent", session_id="run-42", namespace="project-x",
    )

    result = cb.recall(
        "how are failed charges handled?",
        namespace="project-x", token_budget=512,
    )
    print(result["context"])   # budget-bounded, reranked, deduped
    print(result["sources"])   # provenance for every included chunk
```

An `AsyncContextBridgeClient` with the same surface is available for async agents.

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/v1/memory/write` | Chunk, embed, govern and store content |
| `POST` | `/v1/memory/write_batch` | Write many memories in one request |
| `POST` | `/v1/memory/query` | Hybrid recall within a token budget |
| `GET` | `/v1/memory` | List / paginate records by namespace |
| `GET` · `DELETE` | `/v1/memory/{id}` | Fetch / remove a record |
| `POST` | `/v1/memory/summarize` | Compress a session into a summary memory |
| `GET` | `/v1/sessions/{id}/timeline` | Episodic / provenance view |
| `POST` | `/v1/maintenance/sweep` | Delete TTL-expired memories |
| `GET` | `/health` · `/healthz` · `/metrics` | Liveness · readiness · Prometheus |

Interactive OpenAPI docs are served at `/docs`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `QDRANT_URL` | `:memory:` | Qdrant location or `:memory:` |
| `EMBED_PROVIDER` | `hashing` | `hashing` (offline) or `fastembed` |
| `RERANK_PROVIDER` | `identity` | `identity` or `fastembed` |
| `DATABASE_URL` | `sqlite+pysqlite:///./context_bridge.db` | Episodic store |
| `WORKING_PROVIDER` | `memory` | `memory` or `redis` |
| `SUMMARIZER_PROVIDER` | `extractive` | `extractive` or `llm` |
| `DEFAULT_TOKEN_BUDGET` | `2048` | Default recall token budget |
| `API_KEYS` | _(empty)_ | Comma-separated keys; empty = open |
| `API_KEY_NAMESPACES` | _(empty)_ | JSON map of key → allowed namespaces |
| `RATE_LIMIT_PER_MINUTE` | `0` | Per-identity limit; `0` = disabled |
| `SWEEP_INTERVAL_SECONDS` | `0` | Background TTL sweep; `0` = disabled |

See [`.env.example`](.env.example) for the complete list.

## Deployment

```bash
docker build -t context-bridge .
docker run -p 8000:8000 \
  -e QDRANT_URL=http://qdrant:6333 \
  -e DATABASE_URL=postgresql+psycopg://user:pass@postgres/context_bridge \
  context-bridge
```

The image runs as a non-root user and ships a `/health` healthcheck. Apply
database migrations with `alembic upgrade head` (the API also auto-creates
tables for local development).

## Observability

`/metrics` exposes write/query/dedup counters, token-usage and chunk-count
histograms, sweep totals and a request-latency histogram — point Prometheus at
it to quantify the token savings. Every response carries an `X-Request-ID`.

## Development

```bash
make test        # pytest (hermetic: in-memory Qdrant + SQLite)
make lint        # ruff
make typecheck   # mypy
make fmt         # ruff format + autofix
```

See [CONTRIBUTING.md](CONTRIBUTING.md). The test suite needs no network: it uses
in-process Qdrant, SQLite and a deterministic embedder; model-dependent tests
skip automatically when offline.

## Roadmap

- OpenTelemetry tracing
- Grafana dashboard + Prometheus scrape config
- Helm chart / Kubernetes manifests
- Pluggable reran/embedder backends (OpenAI, Cohere) as first-class extras
- Benchmark suite quantifying token savings vs. transcript passing

## License

[Apache-2.0](LICENSE)
