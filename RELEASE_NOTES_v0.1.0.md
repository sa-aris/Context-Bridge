<!--
  ─────────────────────────────────────────────────────────────────────────
  GitHub Release oluştururken:
    • Tag    : v0.1.0   ("Create new tag: v0.1.0 on publish")
    • Title  : Context Bridge 0.1.0
    • Description : AŞAĞIDAKİ ÇİZGİNİN ALTINDAKİ HER ŞEYİ kopyalayıp yapıştır.
    • GIF    : docs/assets/demo.gif dosyasını açıklama kutusunun içine
               SÜRÜKLE-BIRAK (en üste koymak istersen imleci başa al).
  Bu yorum satırlarını (<!-- ... -->) kopyalamana gerek yok.
  ─────────────────────────────────────────────────────────────────────────
-->

**Context Bridge** is shared neural memory middleware for multi-agent systems —
agents write into one governed pool and recall only the task-scoped,
token-budgeted slice they need, instead of passing ever-growing transcripts
around.

### Highlights

**Core**
- Hybrid retrieval (dense + sparse, RRF) → cross-encoder rerank → MMR → token-budgeted assembly
- Three memory tiers: working (in-process / Redis), semantic (Qdrant), episodic / provenance (SQL)
- Governed writes: dedup, confidence gating, TTL decay; small-to-big chunking
- Multi-tenant namespace RBAC, API keys, rate limiting
- Prometheus metrics + Grafana dashboard; typed Python SDK (sync + async)
- Docker image, Helm chart, Alembic migrations

**Cognitive layer (opt-in)**
- Reflective consolidation, contradiction detection & belief revision
- Knowledge graph + ontology alignment
- Failure memory: capture lessons, auto-distill them, and raise them as guardrails before similar work
- Outcome-feedback re-ranking, recall explainability, collaboration-quality score
- Memory health panel, belief timeline, scheduled maintenance, event webhooks, namespace import/export

### Install

```bash
pip install context-bridge-memory
```

### Quick start

```bash
python examples/quickstart.py   # offline, no services required
```

Full docs, the API reference and the live demo are in the README.

Python 3.11+ · Apache-2.0
