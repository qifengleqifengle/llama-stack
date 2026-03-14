import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from .schemas import DocumentChunkInfo, DocumentInfo


class CatalogStore:
    """Simple local registry for ingested document metadata."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"documents": {}, "chunks": {}}, indent=2), encoding="utf-8")

    def _read(self) -> dict:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload.setdefault("documents", {})
        payload.setdefault("chunks", {})
        return payload

    def _write(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_documents(self, vector_db_id: str) -> list[DocumentInfo]:
        with self._lock:
            payload = self._read()
            docs = payload.get("documents", {}).get(vector_db_id, [])
            return [DocumentInfo.model_validate(doc) for doc in docs]

    def add_documents(self, vector_db_id: str, documents: list[DocumentInfo]) -> None:
        with self._lock:
            payload = self._read()
            existing = payload.setdefault("documents", {}).setdefault(vector_db_id, [])
            existing.extend(doc.model_dump() for doc in documents)
            self._write(payload)

    def list_document_chunks(self, vector_db_id: str, document_id: str) -> list[DocumentChunkInfo]:
        with self._lock:
            payload = self._read()
            chunks = payload.get("chunks", {}).get(vector_db_id, {}).get(document_id, [])
            return [DocumentChunkInfo.model_validate(chunk) for chunk in chunks]

    def add_document_chunks(self, vector_db_id: str, document_id: str, chunks: list[DocumentChunkInfo]) -> None:
        with self._lock:
            payload = self._read()
            vector_chunks = payload.setdefault("chunks", {}).setdefault(vector_db_id, {})
            vector_chunks[document_id] = [chunk.model_dump() for chunk in chunks]
            self._write(payload)

    def remove_document(self, vector_db_id: str, document_id: str) -> tuple[DocumentInfo | None, list[DocumentChunkInfo]]:
        with self._lock:
            payload = self._read()

            documents = payload.setdefault("documents", {}).setdefault(vector_db_id, [])
            removed_document: DocumentInfo | None = None
            remaining_documents = []
            for raw_document in documents:
                document = DocumentInfo.model_validate(raw_document)
                if document.document_id == document_id and removed_document is None:
                    removed_document = document
                    continue
                remaining_documents.append(raw_document)

            if remaining_documents:
                payload["documents"][vector_db_id] = remaining_documents
            else:
                payload["documents"].pop(vector_db_id, None)

            vector_chunks = payload.setdefault("chunks", {}).setdefault(vector_db_id, {})
            removed_chunks = [
                DocumentChunkInfo.model_validate(chunk) for chunk in vector_chunks.pop(document_id, [])
            ]
            if not vector_chunks:
                payload["chunks"].pop(vector_db_id, None)

            self._write(payload)
            return removed_document, removed_chunks

    def remove_knowledge_base(self, vector_db_id: str) -> None:
        with self._lock:
            payload = self._read()
            payload.setdefault("documents", {}).pop(vector_db_id, None)
            payload.setdefault("chunks", {}).pop(vector_db_id, None)
            self._write(payload)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(UTC).isoformat()
