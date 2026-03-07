# Agentic RAG with OpenGauss

This guide demonstrates how to run an **agentic Retrieval-Augmented Generation (RAG)** pipeline using
[Llama Stack](https://github.com/meta-llama/llama-stack) with **OpenGauss** as the vector store backend.

---

## Prerequisites

### 1. Running OpenGauss instance

OpenGauss must be running with the **vector** extension enabled.  A quick way to start one locally:

```bash
docker run -d \
  --name opengauss \
  -e GS_PASSWORD=YourPassword123 \
  -p 5432:5432 \
  opengauss/opengauss:latest
```

Verify the vector extension is available:

```sql
-- Connect and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. Running Llama Stack server with OpenGauss provider

Your Llama Stack distribution must include the `remote::opengauss` vector IO provider.  Start the server and
ensure the provider is configured with the correct OpenGauss connection settings.  For example, in your
`run.yaml`:

```yaml
vector_io:
  - provider_id: remote::opengauss
    provider_type: remote::opengauss
    config:
      host: ${env.OPENGAUSS_HOST:=localhost}
      port: ${env.OPENGAUSS_PORT:=5432}
      db: ${env.OPENGAUSS_DB:=postgres}
      user: ${env.OPENGAUSS_USER:=gaussdb}
      password: ${env.OPENGAUSS_PASSWORD}
```

### 3. Python dependencies

```bash
pip install llama-stack-client
```

---

## Running the demo script

Set environment variables and run:

```bash
export LLAMA_STACK_HOST=localhost
export LLAMA_STACK_PORT=8321
export MODEL_ID=meta-llama/Llama-3.2-3B-Instruct

export OPENGAUSS_HOST=localhost
export OPENGAUSS_PORT=5432
export OPENGAUSS_DB=postgres
export OPENGAUSS_USER=gaussdb
export OPENGAUSS_PASSWORD=YourPassword123

python agentic_rag_with_opengauss.py
```

The script will:
1. Connect to the Llama Stack server
2. Register a new vector database backed by `remote::opengauss`
3. Ingest a few sample documents about Python, OpenGauss, RAG, and hybrid search
4. Create an agent with the `builtin::rag` toolgroup configured for **hybrid search with RRF reranking**
5. Ask the agent several questions and print the streamed responses
6. Unregister the vector database on exit

---

## Supported search modes

The `remote::opengauss` provider supports three search modes, configurable via the `query_config` in the
RAG toolgroup arguments:

### Vector search (default)

Uses cosine distance between query embedding and stored embeddings:

```python
"query_config": {"mode": "vector", "max_chunks": 5}
```

### Keyword search

Uses PostgreSQL native full-text search (`tsvector` / `ts_rank`) on the document content:

```python
"query_config": {"mode": "keyword", "max_chunks": 5}
```

### Hybrid search

Combines vector similarity and keyword match scores.  Two reranking strategies are available:

**Reciprocal Rank Fusion (RRF)** – merges ranks from both retrievers; robust to score scale differences:

```python
"query_config": {
    "mode": "hybrid",
    "ranker": {"strategy": "rrf", "params": {"k": 60}},
    "max_chunks": 5,
}
```

**Weighted** – linearly combines min-max normalised scores; `alpha` controls keyword vs. vector weight
(`alpha=0` → pure vector, `alpha=1` → pure keyword, default `alpha=0.5`):

```python
"query_config": {
    "mode": "hybrid",
    "ranker": {"strategy": "weighted", "params": {"alpha": 0.5}},
    "max_chunks": 5,
}
```

---

## How it works

```
User question
      │
      ▼
Llama Stack Agent
      │
      ├─── builtin::rag tool ───▶ OpenGauss vector store
      │        │                        │
      │        │  vector search  ◀──────┤
      │        │  keyword search ◀──────┤
      │        │  hybrid rerank  ◀──────┘
      │        │
      │        └──▶ top-k retrieved chunks (context)
      │
      └─── LLM generates grounded answer
```

The `content_tsv` column is a PostgreSQL **generated column** (populated automatically from the
`document->>'content'` field) indexed with a GIN index for efficient full-text search.  Vector embeddings
are stored in a `vector(N)` column and searched with the `<=>` cosine distance operator provided by the
pgvector-compatible extension in OpenGauss.
