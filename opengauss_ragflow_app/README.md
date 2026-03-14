# OpenGauss RAGFlow App

`opengauss_ragflow_app` is a root-level application layer for the `opengauss-add` workstream.
It is a focused RAG workspace, not a full RAGFlow clone.

## What it includes

- FastAPI backend
- Single-page frontend served by the backend
- Knowledge base creation against `remote::pgvector`
- Document ingestion for pasted text plus MinerU-based extraction for URLs and uploaded files
- Retrieval lab for `vector`, `keyword`, and `hybrid` queries
- Grounded chat using OpenAI-compatible chat completions
- Local document registry for UI visibility

## Structure

```text
opengauss_ragflow_app/
  backend/
  frontend/
  data/
  README.md
```

## Architecture

- Backend: FastAPI
- Frontend: static HTML, CSS, and vanilla JavaScript
- Retrieval: `llama_stack_client`
- Generation: OpenAI-compatible `/v1/chat/completions`
- Extraction: MinerU API
- Knowledge storage: Llama Stack `vector_dbs` + PGVector provider
- App-side metadata: local JSON catalog in `data/catalog.json`

## Required environment

This app assumes:

- A running Llama Stack server
- The server is configured with `remote::pgvector`
- PostgreSQL/PGVector is running with the `vector` extension enabled
- At least one embedding model is available through Llama Stack
- A reachable OpenAI-compatible inference endpoint
- A reachable MinerU API token for URL/file extraction

## Run

From the repository root:

```bash
python3 -m pip install -r opengauss_ragflow_app/requirements.txt
python3 -m uvicorn opengauss_ragflow_app.backend.main:app --env-file opengauss_ragflow_app/.env --reload --port 8787
```

Then open:

```text
http://localhost:8787
```

## Environment

The app reads these environment variables at startup:

- `OPENGAUSS_RAGFLOW_LLAMA_STACK_BASE_URL`
- `OPENGAUSS_RAGFLOW_OPENAI_BASE_URL`
- `OPENGAUSS_RAGFLOW_OPENAI_API_KEY`
- `OPENGAUSS_RAGFLOW_OPENAI_MODEL`
- `OPENGAUSS_RAGFLOW_MINERU_BASE_URL`
- `OPENGAUSS_RAGFLOW_MINERU_API_TOKEN`
- `OPENGAUSS_RAGFLOW_MINERU_MODEL_VERSION`
- `OPENGAUSS_RAGFLOW_MINERU_HTML_MODEL_VERSION`
- `OPENGAUSS_RAGFLOW_MINERU_LANGUAGE`
- `OPENGAUSS_RAGFLOW_DEFAULT_PROVIDER_ID`

For the Llama Stack server side, a separate template is included at:

- `opengauss_ragflow_app/llama_stack.env`
- `opengauss_ragflow_app/llama_stack.env.example`
- `opengauss_ragflow_app/llama_stack_run.yaml`

Example:

```bash
export OPENGAUSS_RAGFLOW_LLAMA_STACK_BASE_URL=http://localhost:8321
export OPENGAUSS_RAGFLOW_OPENAI_BASE_URL=http://localhost:8000/v1
export OPENGAUSS_RAGFLOW_OPENAI_API_KEY=your-key
export OPENGAUSS_RAGFLOW_OPENAI_MODEL=gpt-4o-mini
export OPENGAUSS_RAGFLOW_MINERU_BASE_URL=https://mineru.net
export OPENGAUSS_RAGFLOW_MINERU_API_TOKEN=your-mineru-token
```

## Default wiring

- Llama Stack base URL: `http://localhost:8321`
- OpenAI-compatible base URL: `http://localhost:8000/v1`
- MinerU base URL: `https://mineru.net`
- Default vector provider: `remote::pgvector`
- Default retrieval mode: `hybrid`
- Default reranker: `rrf`

If you want to change these defaults, edit:

- `opengauss_ragflow_app/backend/config.py`

## API summary

- `GET /api/bootstrap`
- `POST /api/knowledge-bases`
- `DELETE /api/knowledge-bases/{vector_db_id}`
- `GET /api/knowledge-bases/{vector_db_id}/documents`
- `POST /api/knowledge-bases/{vector_db_id}/documents/text`
- `POST /api/knowledge-bases/{vector_db_id}/documents/url`
- `POST /api/knowledge-bases/{vector_db_id}/documents/upload`
- `POST /api/retrieval/query`
- `POST /api/chat/sessions`
- `POST /api/chat/sessions/{session_id}/messages`

## Product intent

This app keeps the parts of RAGFlow that matter for your OpenGauss validation:

- knowledge base management
- document ingestion and extraction visibility
- retrieval debugging
- grounded chat with citations

It deliberately skips platform-scale concerns such as multi-tenancy, workflow canvas, and access control.
