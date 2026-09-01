# raggit

**production grade advanced RAG system**

raggit connects directly to local and remote object storage, automatically indexes documents, and answers questions using hybrid retrieval (BM25 + semantic) with reranking and LLM augmentation.

📖 [Full documentation](https://raggit.pages.dev/)

---

## Core Features

- **Automatic continuous indexing** — watch local filesystem or cloud storage and index new or changed documents automatically.
- **Hybrid retrieval** — combine BM25 keyword search with dense semantic search, fused with weighted Reciprocal Rank Fusion.
- **Format-aware chunking** — preserve structure for Markdown, code, PDFs, and plain text with configurable token-based sizing and overlap.
- **Deduplication and cleaning** — remove near-duplicate chunks and normalize whitespace, unicode, and hyphenation.
- **Multi-tenant filtering** — filter by source URI, filename, tenant, tags, document IDs, and date range.
- **Safety and observability** — optional PII redaction, prompt-injection hardening, groundedness checks, and structured audit logging to PostgreSQL.
- **Reranking and parent-window expansion** — cross-encoder reranking and context expansion around top hits.
- **Query rewriting** — optional multi-query and HyDE expansion for better recall.
- **Multiple storage backends** — local filesystem, S3, Google Cloud Storage, and Azure Blob Storage.
- **OpenAI-compatible LLMs** — use OpenAI, Ollama, or any compatible provider for generation and embeddings.
- **FastAPI HTTP API** — query, manage documents and chunks, inspect logs, update configuration, trigger ingestion, and control the watcher over HTTP.
- **Evaluation framework** — run retrieval and answer-quality metrics (recall@k, MRR, NDCG, semantic similarity, LLM-as-judge, groundedness) against annotated datasets via CLI or HTTP API.
- **Optional MCP server** — connect raggit to any MCP client via stdio or SSE after installing the `mcp` extra.

---

## Architecture

```mermaid
flowchart TB
    subgraph Storage
        LS[Local Filesystem]
        RS[S3 / GCS / Azure Blob]
    end

    W[Watcher Service]
    I[Indexing Service]
    P[Parser Registry]
    FA[Format-Aware Chunker]
    DED[Dedup]
    CL[Chunk Cleaner]
    PI[PII Redaction]
    IH[Injection Hardening]
    E[Embedder]
    VS[Qdrant Vector Store]

    W --> I
    I --> P
    P --> FA
    FA --> DED
    DED --> CL
    CL --> PI
    PI --> IH
    IH --> E
    E --> VS

    subgraph PostgreSQL
        D[documents]
        CH[chunks]
        EC[embedding_collections]
        LG[logs]
    end

    I --> D
    IH --> CH
    E --> CH
    I --> EC

    Q[User Query] --> QS[Query Sanitizer]
    QR[Query Rewriter]
    QR --> MQ[Multi-Query]
    QR --> HY[HyDE]

    Q --> QR
    QR --> BM25[Postgres FTS / BM25]
    QR --> SEM[Qdrant Semantic Search]
    BM25 --> RRF[Weighted RRF]
    SEM --> RRF
    RRF --> RR[Cross-Encoder Reranker]
    RR --> TH[Score Threshold]
    TH --> PW[Parent-Window Expansion]
    PW --> AUG[Augmenter]
    AUG --> LLM[LLM Provider]
    LLM --> GRD[Groundedness Check]
    GRD --> OUT[Answer + Citations]

    IH --> LG
    E --> LG
    RRF --> LG
    AUG --> LG
    LLM --> LG
```

### Ingestion Pipeline

1. **Watch** local filesystem or remote object storage for changes.
2. **Parse** PDF, DOCX, HTML, Markdown, and plain text.
3. **Chunk** in a format-aware way (headers, code definitions, page markers, recursive fallback).
4. **Dedup** near-identical chunks using content hash + Jaccard similarity.
5. **Clean** chunks (normalize unicode, collapse whitespace, fix hyphenation).
6. **Redact PII** (optional) and **harden** against prompt-injection patterns.
7. **Embed** chunks using local sentence-transformers or an OpenAI-compatible API.
8. **Store** vectors in Qdrant and metadata/links in PostgreSQL.
9. **Audit** every ingestion input/output to the Postgres `logs` table.

### Retrieval Pipeline

1. **Sanitize** and optionally **rewrite** the query (multi-query / HyDE).
2. **Search** with BM25 via PostgreSQL and semantic search via Qdrant.
3. **Fuse** ranked lists with weighted Reciprocal Rank Fusion.
4. **Rerank** top-N candidates with an optional cross-encoder.
5. **Threshold** low-confidence chunks and expand parent-document windows.
6. **Augment** the prompt with isolated context and generate an answer via an LLM.
7. **Check** groundedness and **cite** every chunk with source URI, filename, page, and score.
8. **Audit** the query, retrieval result, and final answer to the Postgres `logs` table.

---

## Setup

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (for Docker deployment)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (for local development)

### Docker (recommended)

Build and run the entire stack:

```bash
   docker compose up -d
   ```

   This starts:

   - `raggit-postgres` on port `5433`
   - `raggit-qdrant` on ports `6333`/`6334`
   - `raggit-app` running the FastAPI server and watcher/indexer service on port `8000`

   If you rebuild the image after Dockerfile changes, recreate the app container:

```bash
docker compose down raggit
docker compose build --no-cache raggit
docker compose up -d raggit
```

Place documents in `./data/documents` for local storage ingestion.

### Local development

1. Start PostgreSQL and Qdrant:

```bash
docker compose up -d postgres qdrant
```

2. Install dependencies:

```bash
uv sync
```

3. Run migrations:

```bash
uv run alembic upgrade head
```

4. Configure raggit:

```bash
uv run raggit setup \
  --database-url postgresql+asyncpg://raggit:raggit@localhost:5433/raggit \
  --qdrant-url http://localhost:6333 \
  --storage-source-type local \
  --storage-uri ./data/documents \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-api-key $OPENAI_API_KEY
```

5. Ingest documents:

```bash
uv run raggit ingest ./data/documents
```

6. Query:

```bash
uv run raggit query "What is raggit?"
```

7. Run continuous indexing:

```bash
uv run raggit serve
```

   The `serve` command also starts the FastAPI HTTP server on port 8000. Visit `/docs` for interactive Swagger UI.

8. **Use the HTTP API**:

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/status
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is raggit?"}'
   ```

9. **Evaluate the system** (three tiers – component → pipeline → system):

   ```bash
   # Component: isolated primitive (parser, chunker, pii, sanitizer, RRF, etc.)
   uv run raggit eval --generate --kind component --component chunker --name chunker-suite
   uv run raggit eval chunker-suite.yaml

   # Pipeline: ingestion and retrieval chains
   uv run raggit eval --generate --kind pipeline --pipeline ingestion --name pipe
   uv run raggit eval pipe.yaml

   # System: end-to-end (default)
   uv run raggit eval --generate --name my-eval
   uv run raggit eval my-eval.yaml --output report.json

   # Comprehensive (every feature) or all tiers at once
   uv run raggit eval --comprehensive --name full-suite
   uv run raggit eval --generate --kind all --name all-tiers
   uv run raggit eval --list-metrics   # 69+ metrics
   ```

   Component tier covers parser, chunker, cleaner, PII, injection, sanitizer, embedder, RRF, reranker, safety, storage, watcher, retriever. Pipeline tier covers ingestion (parse→chunk→clean→embed) and retrieval (sanitize→rewrite→BM25/semantic→RRF→rerank→threshold→parent→traversal). System tier covers retrieval + answer quality, groundedness, citations, hallucination, tenant/tag/prefix filters, latency p50/p95, audit, MCP.

10. **Connect via MCP (optional)**:

    Install the `mcp` extra and start the MCP server:

    ```bash
    uv pip install 'raggit[mcp]'
    uv run raggit mcp
    ```

    The MCP server is also mounted at `/mcp` on the FastAPI HTTP server when the `mcp` extra is installed. It exposes tools for query, status, documents, chunks, logs, ingestion, and evaluation.

    See the [full documentation](https://raggit.pages.dev/) for cloud storage options, configuration reference, CLI details, API reference, and MCP usage.
