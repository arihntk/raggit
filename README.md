# raggit

**Plug-and-play production-grade RAG system**

raggit connects directly to local and remote object storage, automatically indexes documents, and answers questions using hybrid retrieval (BM25 + semantic) with reranking and LLM augmentation.

**Full documentation:** https://raggit.pages.dev/

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
2. **Parse** PDF, DOCX, HTML, Markdown, and plain text. PDF pages are marked explicitly so page numbers survive chunking.
3. **Chunk** in a format-aware way:
   - Markdown by headers
   - Code by top-level definitions
   - PDF by page markers
   - Fallback recursive character splitting
4. **Size** chunks by tokens (`tiktoken` cl100k_base when available) with configurable overlap.
5. **Dedup** near-identical chunks using content hash + Jaccard word-set similarity.
6. **Clean** chunks (normalize unicode, collapse whitespace, fix hyphenation).
7. **Redact PII** (optional) and **harden** against prompt-injection patterns before embedding.
8. **Embed** chunks using local sentence-transformers or an OpenAI-compatible API.
9. **Store** vectors in a model-scoped Qdrant collection and metadata/links in PostgreSQL.
10. **Audit** every ingestion input/output to the Postgres `logs` table.

### Retrieval Pipeline

1. **Sanitize** the query to extract keywords.
2. **Rewrite** the query when configured:
   - `multi_query`: generate alternative phrasings for broader recall
   - `hyde`: generate a hypothetical answer passage to embed
3. **Filter** BM25 and semantic searches by metadata: source URI prefix, filename prefix, tenant, tags, document ids, date range.
4. **BM25** keyword search via PostgreSQL full-text search (`tsvector` GIN index).
5. **Semantic** similarity search via Qdrant cosine distance.
6. **Fuse** ranked lists with weighted Reciprocal Rank Fusion (tunable BM25/semantic weights).
7. **Rerank** top-N candidates with an optional cross-encoder (e.g. `BAAI/bge-reranker-base`).
8. **Threshold** low-confidence chunks with `min_score` and refuse empty/low-score retrieval.
9. **Expand** parent-document windows around each hit when `parent_window > 0`.
10. **Cite** every chunk with id, source URI, filename, page, section, offsets, score, and excerpt.
11. **Augment** the prompt with isolated untrusted-context wrappers and system instructions.
12. **Generate** an answer via OpenAI-compatible API or Ollama.
13. **Check** groundedness and warn if the answer is not supported by retrieved context.
14. **Audit** the query, retrieval result, and final answer to the Postgres `logs` table.

---

## Tech Stack

- **Runtime:** Python 3.14 (managed by `uv`)
- **Database:** PostgreSQL 16+ (metadata, chunks, logs)
- **Vector Store:** Qdrant
- **Infra:** Docker Compose
- **CLI:** `typer` + `rich` (progress bars, panels, citation trees)
- **Logging:** `structlog` console logs + Postgres `logs` table for structured audit events
- **Storage:** Local filesystem, S3, GCS, Azure Blob (optional extras)
- **ORM/Migrations:** SQLAlchemy 2.0 + Alembic

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine + Compose)
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### 1. Clone and enter the project

```bash
cd raggit
```

### 2. Start PostgreSQL and Qdrant

```bash
docker compose up -d postgres qdrant
```

> If port `5432` is already in use, the compose file maps PostgreSQL to `5433:5432` by default. Update `DATABASE_URL` accordingly.

### 3. Install dependencies

```bash
uv sync
```

### 4. Run database migrations

```bash
uv run alembic upgrade head
```

### 5. Configure raggit

#### Local storage (default)

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

#### AWS S3

Install the S3 extra first:

```bash
uv pip install 'raggit[s3]'
```

Then run setup:

```bash
uv run raggit setup \
  --database-url postgresql+asyncpg://raggit:raggit@localhost:5433/raggit \
  --qdrant-url http://localhost:6333 \
  --storage-source-type s3 \
  --storage-uri s3://my-bucket/documents \
  --storage-bucket my-bucket \
  --storage-prefix documents \
  --storage-region us-east-1 \
  --aws-access-key-id $AWS_ACCESS_KEY_ID \
  --aws-secret-access-key $AWS_SECRET_ACCESS_KEY \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-api-key $OPENAI_API_KEY
```

#### Google Cloud Storage

Install the GCS extra first:

```bash
uv pip install 'raggit[gcs]'
```

Then run setup:

```bash
uv run raggit setup \
  --database-url postgresql+asyncpg://raggit:raggit@localhost:5433/raggit \
  --qdrant-url http://localhost:6333 \
  --storage-source-type gcs \
  --storage-uri gs://my-bucket/documents \
  --storage-bucket my-bucket \
  --storage-prefix documents \
  --gcs-service-account-path /path/to/service-account.json \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-api-key $OPENAI_API_KEY
```

#### Azure Blob Storage

Install the Azure extra first:

```bash
uv pip install 'raggit[azure]'
```

Then run setup:

```bash
uv run raggit setup \
  --database-url postgresql+asyncpg://raggit:raggit@localhost:5433/raggit \
  --qdrant-url http://localhost:6333 \
  --storage-source-type azure_blob \
  --storage-uri azure://my-container/documents \
  --storage-container my-container \
  --storage-prefix documents \
  --azure-connection-string $AZURE_STORAGE_CONNECTION_STRING \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-api-key $OPENAI_API_KEY
```

### 6. Add documents and ingest

For local storage:

```bash
mkdir -p data/documents
cp my-docs/*.pdf data/documents/
uv run raggit ingest ./data/documents
```

For cloud storage, upload documents to the configured bucket/container and run:

```bash
uv run raggit ingest
```

### 7. Ask questions

```bash
uv run raggit query "What is raggit?"
```

### 8. Run the watcher (continuous indexing)

```bash
uv run raggit watch
```

For local storage you can override the path:

```bash
uv run raggit watch ./data/documents
```

---

## CLI Commands

| Command | Description |
|---|---|
| `raggit setup` | Interactive configuration (local, S3, GCS, Azure) |
| `raggit ingest <path>` | One-time ingestion with a progress bar (path is optional for cloud storage) |
| `raggit watch <path>` | Continuously watch and index with live event indicators |
| `raggit query "<question>"` | Ask a question; shows status spinners, chunk table, answer panel, and citation tree |
| `raggit status` | Show indexed document status and active embedding collections |

Query supports `--top-k`, `--source-prefix`, `--filename-prefix`, `--tenant`, `--tag`, `--min-score`, and `--rewrite`.

---

## Docker Deployment

Build and run the entire stack:

```bash
docker compose up -d
```

This starts:

- `raggit-postgres` on port `5433`
- `raggit-qdrant` on ports `6333`/`6334`
- `raggit-app` running the watcher

Mount your documents into `./data/documents`.

---

## Development

```bash
# Run linting
uv run ruff check .

# Run type checking
uv run mypy src

# Run tests
uv run pytest
```

---

## Configuration

Configuration is loaded from environment variables and a `~/.config/raggit/raggit.env` file generated by `raggit setup`.

Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://raggit:raggit@localhost:5433/raggit` | PostgreSQL connection |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant URL |
| `QDRANT_COLLECTION` | `raggit_chunks` | Qdrant collection name |
| `STORAGE_SOURCE_TYPE` | `local` | Storage backend: `local`, `s3`, `gcs`, `azure_blob` |
| `STORAGE_URI` | `./data/documents` | Storage URI or local path |
| `STORAGE_BUCKET` | `None` | S3/GCS bucket name |
| `STORAGE_CONTAINER` | `None` | Azure container name |
| `STORAGE_PREFIX` | `None` | Object prefix / folder |
| `STORAGE_REGION` | `None` | S3 region |
| `STORAGE_AWS_ACCESS_KEY_ID` | `None` | AWS access key ID |
| `STORAGE_AWS_SECRET_ACCESS_KEY` | `None` | AWS secret access key |
| `STORAGE_GCS_SERVICE_ACCOUNT_PATH` | `None` | GCS service account JSON path |
| `STORAGE_AZURE_CONNECTION_STRING` | `None` | Azure Blob connection string |
| `LLM_PROVIDER` | `openai` | LLM provider |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_API_KEY` | `None` | API key |
| `EMBEDDING_PROVIDER` | `sentence-transformers` | Embedding provider |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |

---

## License

MIT
