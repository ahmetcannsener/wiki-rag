# Production Deployment Recommendation

## Current Limitations

The localhost setup works well as a prototype but has several hard limits that prevent production use:

| Limitation | Impact |
|---|---|
| Single-process Streamlit | One user at a time; no concurrent requests |
| Ollama on local CPU/GPU | Inference is slow (10–30 s/query); not horizontally scalable |
| ChromaDB on local disk | No replication, no multi-node access, data lost if disk fails |
| SQLite | File-level locking; breaks under concurrent writes |
| No authentication | Anyone on the network can access the app |
| No monitoring | No visibility into errors, latency, or usage |

---

## Recommended Production Stack

### 1. Language Model → Cloud LLM API

Replace `ollama/llama3.2` with a managed API:

- **Anthropic Claude** (`claude-sonnet-4-6`) — strong instruction-following, long context, grounded responses with citations
- **OpenAI GPT-4o** — broad knowledge, reliable JSON output for structured use cases

Both offer sub-3-second latency at scale and eliminate the need to manage GPU infrastructure. Use prompt caching (Anthropic) or response caching (a Redis layer) to reduce cost on repeated or similar queries.

**Tradeoff:** data leaves your infrastructure. If the Wikipedia content is sensitive or the deployment is air-gapped, keep a self-hosted model (vLLM + Llama 3 on a GPU server) instead.

### 2. Embeddings → Managed Embedding API

Replace `nomic-embed-text` via Ollama with:

- **OpenAI `text-embedding-3-small`** — fast, cheap, 1536-dim vectors, strong retrieval quality
- **Voyage AI `voyage-3`** — state-of-the-art retrieval benchmarks, drop-in REST API

Re-index all chunks once at migration time. Store the embedding model name in metadata so re-indexing with a different model is detectable.

### 3. Vector Store → Managed Vector Database

Replace ChromaDB (local) with:

- **Pinecone** — fully managed, serverless, automatic scaling, built-in metadata filtering
- **Weaviate Cloud** — open-source core with managed hosting, supports hybrid BM25 + vector search out of the box

Both support the same upsert/query pattern used in `vector_store.py`. Migration requires a one-time re-index and a client swap (`chromadb` → `pinecone-client` or `weaviate-client`).

### 4. Relational Database → PostgreSQL

Replace `wikipedia.db` (SQLite) with **PostgreSQL**:

- Handles concurrent reads and writes
- Add `pgvector` extension if you want SQL-level similarity search as a fallback
- Use `psycopg3` or `asyncpg`; schema is a single table so migration is trivial

### 5. Application Layer → FastAPI + React (or hosted Streamlit)

Two options depending on audience:

**Option A — Internal / demo tool:** Deploy Streamlit on **Streamlit Community Cloud** or **AWS App Runner**. Zero infrastructure change, just point at a repo. Add `st.secrets` for API keys.

**Option B — Customer-facing product:** Replace Streamlit with a **FastAPI** backend and a **React** frontend. FastAPI gives you async endpoints, proper HTTP status codes, and an OpenAPI spec for free. Deploy on **AWS ECS** or **GCP Cloud Run** behind an **ALB/Cloud Load Balancer**.

### 6. Authentication & Rate Limiting

- Add **OAuth 2.0 / OIDC** (Auth0, Cognito, or Google) in front of the app
- Rate-limit per user at the API gateway layer (AWS API Gateway or Nginx)
- Store user sessions in **Redis** (also useful for response caching)

### 7. Monitoring & Observability

- **LLM call logging:** log prompt, model, latency, token count, and answer to a structured store (CloudWatch, Datadog, or a simple Postgres table)
- **Retrieval quality:** log retrieved chunk titles and distances per query — this is the fastest way to detect degraded retrieval
- **Alerts:** set latency and error-rate thresholds; page on-call if p95 latency exceeds 10 s

---

## Cost vs Privacy Tradeoff

| Dimension | Cloud LLM | Self-hosted LLM |
|---|---|---|
| Latency | Low (1–3 s) | Medium–high (5–30 s without high-end GPU) |
| Cost | Pay-per-token (~$1–5 / 1M tokens) | Fixed hardware + ops cost |
| Privacy | Data sent to third party | Data stays on-premises |
| Maintenance | None | GPU drivers, model updates, scaling |

For most academic or internal projects, the cloud path is the right default. For regulated industries (healthcare, legal, finance) or air-gapped environments, self-hosted on a GPU server with vLLM is the correct choice.

---

## Migration Path (Suggested Order)

1. Swap SQLite → PostgreSQL (low risk, schema unchanged)
2. Re-index with a managed embedding API (one-time batch job)
3. Swap ChromaDB → Pinecone/Weaviate (update `vector_store.py` only)
4. Swap Ollama → Claude/GPT-4o API (update `embedder.py` and `generator.py`)
5. Add authentication layer
6. Deploy on cloud (App Runner / Cloud Run)
7. Wire up monitoring
