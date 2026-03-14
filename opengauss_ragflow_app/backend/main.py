from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .catalog import CatalogStore
from .config import FRONTEND_DIR, load_settings
from .schemas import (
    BootstrapResponse,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionInfo,
    CreateChatSessionRequest,
    CreateKnowledgeBaseRequest,
    DocumentChunkInfo,
    DocumentInfo,
    KnowledgeBaseInfo,
    RetrievalRequest,
    RetrievalResponse,
    TextIngestRequest,
    UrlIngestRequest,
)
from .service import AppGateway


settings = load_settings()
catalog = CatalogStore(settings.catalog_path)
gateway = AppGateway(settings, catalog)

app = FastAPI(title="OpenGauss RAGFlow App", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


def _as_http_error(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(FRONTEND_DIR) / "index.html")


@app.get("/api/bootstrap", response_model=BootstrapResponse)
def bootstrap() -> BootstrapResponse:
    try:
        return gateway.bootstrap()
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.post("/api/knowledge-bases", response_model=KnowledgeBaseInfo)
def create_knowledge_base(request: CreateKnowledgeBaseRequest) -> KnowledgeBaseInfo:
    try:
        return gateway.create_knowledge_base(request)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.delete("/api/knowledge-bases/{vector_db_id}")
def delete_knowledge_base(vector_db_id: str) -> dict[str, str]:
    try:
        gateway.delete_knowledge_base(vector_db_id)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc
    return {"status": "deleted", "vector_db_id": vector_db_id}


@app.get("/api/knowledge-bases/{vector_db_id}/documents", response_model=list[DocumentInfo])
def list_documents(vector_db_id: str) -> list[DocumentInfo]:
    return gateway.list_documents(vector_db_id)


@app.get("/api/knowledge-bases/{vector_db_id}/documents/{document_id}/chunks", response_model=list[DocumentChunkInfo])
def list_document_chunks(vector_db_id: str, document_id: str) -> list[DocumentChunkInfo]:
    return gateway.list_document_chunks(vector_db_id, document_id)


@app.delete("/api/knowledge-bases/{vector_db_id}/documents/{document_id}")
def delete_document(vector_db_id: str, document_id: str) -> dict[str, str]:
    try:
        gateway.delete_document(vector_db_id, document_id)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc
    return {"status": "deleted", "vector_db_id": vector_db_id, "document_id": document_id}


@app.post("/api/knowledge-bases/{vector_db_id}/documents/text", response_model=list[DocumentInfo])
def ingest_text_documents(vector_db_id: str, request: TextIngestRequest) -> list[DocumentInfo]:
    try:
        return gateway.ingest_text_documents(vector_db_id, request.documents, request.chunk_size_in_tokens)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.post("/api/knowledge-bases/{vector_db_id}/documents/url", response_model=list[DocumentInfo])
def ingest_url_documents(vector_db_id: str, request: UrlIngestRequest) -> list[DocumentInfo]:
    try:
        return gateway.ingest_urls(vector_db_id, request.urls, request.chunk_size_in_tokens)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.post("/api/knowledge-bases/{vector_db_id}/documents/upload", response_model=list[DocumentInfo])
def ingest_upload_documents(
    vector_db_id: str,
    chunk_size_in_tokens: int = 512,
    files: list[UploadFile] = File(...),
) -> list[DocumentInfo]:
    try:
        return gateway.ingest_uploads(vector_db_id, files, chunk_size_in_tokens)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.post("/api/retrieval/query", response_model=RetrievalResponse)
def retrieval_query(request: RetrievalRequest) -> RetrievalResponse:
    try:
        return gateway.retrieve(request)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.post("/api/chat/sessions", response_model=ChatSessionInfo)
def create_chat_session(request: CreateChatSessionRequest) -> ChatSessionInfo:
    try:
        return gateway.create_chat_session(request)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc


@app.post("/api/chat/sessions/{session_id}/messages", response_model=ChatMessageResponse)
def send_chat_message(session_id: str, request: ChatMessageRequest) -> ChatMessageResponse:
    try:
        return gateway.send_chat_message(session_id, request.message)
    except Exception as exc:  # pragma: no cover - depends on external server
        raise _as_http_error(exc) from exc
