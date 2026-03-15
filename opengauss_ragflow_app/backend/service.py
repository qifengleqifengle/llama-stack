import io
import importlib
import mimetypes
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import UploadFile
from llama_stack_client import LlamaStackClient

from .catalog import CatalogStore
from .config import AppSettings
from .schemas import (
    BootstrapResponse,
    ChatMessageResponse,
    ChatSessionInfo,
    ChunkingStrategy,
    ChunkResult,
    CitationInfo,
    CreateChatSessionRequest,
    CreateKnowledgeBaseRequest,
    DocumentChunkInfo,
    DocumentInfo,
    KnowledgeBaseInfo,
    ModelInfo,
    RetrievalRequest,
    RetrievalResponse,
    TextDocumentInput,
)


_PSYCOPG2 = None


def _load_psycopg2():
    global _PSYCOPG2
    if _PSYCOPG2 is not None:
        return _PSYCOPG2

    try:
        _PSYCOPG2 = importlib.import_module("psycopg2")
        return _PSYCOPG2
    except ModuleNotFoundError:
        pass

    stack_site_packages = (
        Path(__file__).resolve().parents[2]
        / ".venv-stack"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if stack_site_packages.exists():
        sys.path.append(str(stack_site_packages))
        _PSYCOPG2 = importlib.import_module("psycopg2")
        return _PSYCOPG2

    raise RuntimeError("psycopg2 is unavailable; document deletion requires the stack virtualenv dependencies")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "kb"


def _make_sql_identifier(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _source_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name.strip()
    return name or url


def _join_openai_url(base_url: str, path: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = path.lstrip("/")
    if normalized.endswith("/v1"):
        return f"{normalized}/{suffix}"
    return f"{normalized}/v1/{suffix}"


def _text_preview(value: str, limit: int = 160) -> str:
    collapsed = " ".join(value.split())
    return collapsed[:limit]


def _estimate_chunk_units(value: str) -> int:
    units = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+", value)
    return len(units) if units else len(value.strip())


def _extract_citation_indices(answer: str, max_index: int) -> list[int]:
    indices: list[int] = []
    for raw_match in re.findall(r"\[(\d+)\]", answer or ""):
        index = int(raw_match)
        if 1 <= index <= max_index and index not in indices:
            indices.append(index)
    return indices


def _normalize_chunking_strategy(value: str | None) -> ChunkingStrategy:
    allowed: set[str] = {"mineru_markdown", "fixed_tokens", "fixed_chars", "recursive"}
    normalized = (value or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"Unsupported chunking strategy: {value}")
    return normalized  # type: ignore[return-value]


def _chunk_plain_text(value: str, chunk_size_in_tokens: int) -> list[str]:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return []

    words = normalized.split(" ")
    if len(words) <= chunk_size_in_tokens:
        return [normalized]

    chunks: list[str] = []
    step = max(1, chunk_size_in_tokens)
    for index in range(0, len(words), step):
        chunk = " ".join(words[index : index + step]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _chunk_fixed_chars(value: str, chunk_size_in_chars: int) -> list[str]:
    normalized = value.strip()
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        chunk = normalized[start : start + chunk_size_in_chars].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size_in_chars
    return chunks


def _split_markdown_blocks(value: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for raw_line in value.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        is_heading = bool(re.match(r"^#{1,6}\s+\S+", stripped))

        if is_heading:
            if current:
                block = "\n".join(current).strip()
                if block:
                    blocks.append(block)
                current = []
            current.append(stripped)
            continue

        if not stripped:
            if current:
                block = "\n".join(current).strip()
                if block:
                    blocks.append(block)
                current = []
            continue

        current.append(stripped)

    if current:
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)

    return blocks


def _chunk_markdown(value: str, chunk_size_in_tokens: int) -> list[str]:
    blocks = _split_markdown_blocks(value)
    if not blocks:
        return _chunk_plain_text(value, chunk_size_in_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_units = 0

    for block in blocks:
        block_units = _estimate_chunk_units(block)
        if block_units > chunk_size_in_tokens:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_units = 0
            chunks.extend(_chunk_plain_text(block, chunk_size_in_tokens))
            continue

        if current and current_units + block_units > chunk_size_in_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [block]
            current_units = block_units
            continue

        current.append(block)
        current_units += block_units

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _split_recursive_blocks(value: str) -> list[str]:
    normalized = value.strip()
    if not normalized:
        return []

    paragraph_blocks = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    if len(paragraph_blocks) > 1:
        return paragraph_blocks

    sentence_blocks = [part.strip() for part in re.split(r"(?<=[。！？!?\.])\s+", normalized) if part.strip()]
    if len(sentence_blocks) > 1:
        return sentence_blocks

    return [normalized]


def _chunk_recursive(value: str, chunk_size_in_tokens: int, chunk_size_in_chars: int) -> list[str]:
    blocks = _split_markdown_blocks(value) if "#" in value else _split_recursive_blocks(value)
    if not blocks:
        return _chunk_plain_text(value, chunk_size_in_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_units = 0

    for block in blocks:
        block_units = _estimate_chunk_units(block)
        if block_units > chunk_size_in_tokens or len(block) > chunk_size_in_chars:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_units = 0
            if len(block) > chunk_size_in_chars:
                chunks.extend(_chunk_fixed_chars(block, chunk_size_in_chars))
            else:
                chunks.extend(_chunk_plain_text(block, chunk_size_in_tokens))
            continue

        if current and current_units + block_units > chunk_size_in_tokens:
            chunks.append("\n\n".join(current).strip())
            current = [block]
            current_units = block_units
            continue

        current.append(block)
        current_units += block_units

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def _chunk_document_content(
    document: "PreparedDocument",
    chunking_strategy: ChunkingStrategy,
    chunk_size_in_tokens: int,
    chunk_size_in_chars: int,
) -> tuple[list[str], str]:
    is_mineru_markdown = document.mime_type == "text/markdown" and document.metadata.get("mineru_backend") == "api"

    if chunking_strategy == "mineru_markdown":
        if is_mineru_markdown:
            return _chunk_markdown(document.content, chunk_size_in_tokens), "mineru_markdown"
        return _chunk_recursive(document.content, chunk_size_in_tokens, chunk_size_in_chars), "recursive"

    if chunking_strategy == "fixed_chars":
        return _chunk_fixed_chars(document.content, chunk_size_in_chars), "fixed_chars"

    if chunking_strategy == "recursive":
        return _chunk_recursive(document.content, chunk_size_in_tokens, chunk_size_in_chars), "recursive"

    return _chunk_plain_text(document.content, chunk_size_in_tokens), "fixed_tokens"


def _coerce_openai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content or "")


def _normalize_mineru_state(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", "-")


def _is_terminal_mineru_state(state: str) -> bool:
    return state in {"done", "success", "failed", "error"}


def _detect_mineru_model_version(default_version: str, html_version: str, source_name: str) -> str:
    suffix = Path(source_name.lower()).suffix
    if suffix in {".html", ".htm"}:
        return html_version
    return default_version


@dataclass
class PreparedDocument:
    document_id: str
    title: str
    content: str
    source_type: str
    source_value: str
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSessionState:
    config: ChatSessionInfo
    instructions: str
    alpha: float
    impact_factor: float
    history: list[dict[str, str]] = field(default_factory=list)


class OpenAICompatibleClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def _timeout(self) -> httpx.Timeout:
        total = self.settings.openai_request_timeout_seconds
        return httpx.Timeout(
            timeout=total,
            connect=min(10.0, total),
            read=total,
            write=min(30.0, total),
            pool=min(10.0, total),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"
        return headers

    def list_models(self) -> tuple[list[str], bool]:
        fallback_models = [self.settings.openai_model] if self.settings.openai_model else []
        try:
            with httpx.Client(timeout=self._timeout(), trust_env=False) as client:
                response = client.get(_join_openai_url(self.settings.openai_base_url, "models"), headers=self._headers())
                response.raise_for_status()
            payload = response.json()
            model_ids = [str(item.get("id")) for item in payload.get("data", []) if item.get("id")]
            if not model_ids:
                return fallback_models, bool(fallback_models)
            if self.settings.openai_model and self.settings.openai_model not in model_ids:
                model_ids.insert(0, self.settings.openai_model)
            return list(dict.fromkeys(model_ids)), True
        except Exception:
            return fallback_models, bool(fallback_models)

    def chat_completion(self, model_id: str, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": self.settings.openai_temperature,
            "max_tokens": self.settings.openai_max_tokens,
            "stream": False,
        }
        with httpx.Client(timeout=self._timeout(), trust_env=False) as client:
            response = client.post(
                _join_openai_url(self.settings.openai_base_url, "chat/completions"),
                headers=self._headers(),
                json=payload,
            )
            response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("OpenAI-compatible API returned no choices")
        message = choices[0].get("message", {})
        return _coerce_openai_content(message.get("content")).strip()

    def rewrite_query(self, model_id: str, query: str) -> str:
        prompt = (
            "Rewrite the user's query into one concise retrieval query.\n"
            "Rules:\n"
            "- Preserve the original intent.\n"
            "- Keep named entities, product names, versions, and technical terms.\n"
            "- Do not answer the question.\n"
            "- Do not output multiple queries.\n"
            "- Output only the rewritten query.\n\n"
            f"User query:\n{query}"
        )
        rewritten = self.chat_completion(
            model_id,
            [
                {
                    "role": "system",
                    "content": "You rewrite user questions for retrieval. Output only one rewritten query.",
                },
                {"role": "user", "content": prompt},
            ],
        ).strip()
        return rewritten or query


class MinerUClient:
    def __init__(self, settings: AppSettings):
        self.settings = settings

    def _client(self) -> httpx.Client:
        return httpx.Client(
            timeout=self.settings.mineru_request_timeout_seconds,
            follow_redirects=True,
            trust_env=False,
        )

    def _ensure_configured(self) -> None:
        if not self.settings.mineru_api_token:
            raise ValueError("MinerU API token is not configured")

    def _headers(self) -> dict[str, str]:
        self._ensure_configured()
        return {"Authorization": f"Bearer {self.settings.mineru_api_token}"}

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        with self._client() as client:
            response = client.request(
                method,
                f"{self.settings.mineru_base_url.rstrip('/')}{path}",
                headers={**self._headers(), **kwargs.pop('headers', {})},
                **kwargs,
            )
            response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code") not in (None, 0, 200):
            raise ValueError(payload.get("msg") or payload.get("message") or "MinerU API request failed")
        return payload

    def _download_markdown(self, zip_url: str) -> str:
        with self._client() as client:
            response = client.get(zip_url)
            response.raise_for_status()
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        names = archive.namelist()
        preferred = next((name for name in names if name.endswith("full.md")), None)
        markdown_name = preferred or next((name for name in names if name.endswith(".md")), None)
        if markdown_name is None:
            raise ValueError("MinerU result archive does not contain markdown output")
        return archive.read(markdown_name).decode("utf-8", errors="ignore").strip()

    def _poll_single_task(self, task_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.settings.mineru_poll_timeout_seconds
        while time.monotonic() < deadline:
            payload = self._request_json("GET", f"/api/v4/extract/task/{task_id}")
            result = payload.get("data") or {}
            state = _normalize_mineru_state(result.get("state"))
            if _is_terminal_mineru_state(state):
                if state in {"failed", "error"}:
                    raise ValueError(result.get("err_msg") or result.get("message") or f"MinerU task {task_id} failed")
                return result
            time.sleep(self.settings.mineru_poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for MinerU task {task_id}")

    def _poll_batch_task(self, batch_id: str) -> list[dict[str, Any]]:
        deadline = time.monotonic() + self.settings.mineru_poll_timeout_seconds
        while time.monotonic() < deadline:
            payload = self._request_json("GET", f"/api/v4/extract-results/batch/{batch_id}")
            results = (payload.get("data") or {}).get("extract_result") or []
            if results and all(_is_terminal_mineru_state(_normalize_mineru_state(item.get("state"))) for item in results):
                failures = [
                    item.get("file_name") or item.get("data_id") or "unknown"
                    for item in results
                    if _normalize_mineru_state(item.get("state")) in {"failed", "error"}
                ]
                if failures:
                    raise ValueError(f"MinerU batch extraction failed for: {', '.join(failures)}")
                return results
            time.sleep(self.settings.mineru_poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for MinerU batch {batch_id}")

    def extract_urls(self, urls: list[str]) -> list[PreparedDocument]:
        documents: list[PreparedDocument] = []
        for url in urls:
            document_id = f"url_{uuid4().hex[:10]}"
            title = _source_title_from_url(url)
            payload = {
                "url": url,
                "is_ocr": self.settings.mineru_enable_ocr,
                "enable_formula": self.settings.mineru_enable_formula,
                "enable_table": self.settings.mineru_enable_table,
                "language": self.settings.mineru_language,
                "model_version": _detect_mineru_model_version(
                    self.settings.mineru_model_version,
                    self.settings.mineru_html_model_version,
                    title,
                ),
            }
            task_payload = self._request_json("POST", "/api/v4/extract/task", json=payload)
            task_id = (task_payload.get("data") or {}).get("task_id")
            if not task_id:
                raise ValueError("MinerU did not return a task_id for URL extraction")
            result = self._poll_single_task(task_id)
            zip_url = result.get("full_zip_url")
            if not zip_url:
                raise ValueError(f"MinerU task {task_id} finished without a full_zip_url")
            content = self._download_markdown(zip_url)
            documents.append(
                PreparedDocument(
                    document_id=document_id,
                    title=title,
                    content=content,
                    source_type="url",
                    source_value=url,
                    mime_type="text/markdown",
                    metadata={
                        "source_url": url,
                        "mineru_task_id": task_id,
                        "mineru_backend": "api",
                        "mineru_model_version": payload["model_version"],
                    },
                )
            )
        return documents

    def extract_uploads(self, files: list[UploadFile]) -> list[PreparedDocument]:
        grouped_files: dict[str, list[dict[str, Any]]] = {}
        for upload in files:
            filename = upload.filename or f"upload_{uuid4().hex[:8]}"
            payload = upload.file.read()
            if not payload:
                raise ValueError(f"Uploaded file {filename} is empty")
            document_id = f"file_{uuid4().hex[:10]}"
            mime_type = upload.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
            model_version = _detect_mineru_model_version(
                self.settings.mineru_model_version,
                self.settings.mineru_html_model_version,
                filename,
            )
            grouped_files.setdefault(model_version, []).append(
                {
                    "document_id": document_id,
                    "filename": filename,
                    "payload": payload,
                    "mime_type": mime_type,
                    "title": filename,
                }
            )

        documents: list[PreparedDocument] = []
        for model_version, entries in grouped_files.items():
            request_payload = {
                "enable_formula": self.settings.mineru_enable_formula,
                "enable_table": self.settings.mineru_enable_table,
                "language": self.settings.mineru_language,
                "model_version": model_version,
                "files": [
                    {
                        "name": entry["filename"],
                        "data_id": entry["document_id"],
                        "is_ocr": self.settings.mineru_enable_ocr,
                    }
                    for entry in entries
                ],
            }
            upload_payload = self._request_json("POST", "/api/v4/file-urls/batch", json=request_payload)
            data = upload_payload.get("data") or {}
            batch_id = data.get("batch_id")
            upload_urls = data.get("file_urls") or []
            if not batch_id or len(upload_urls) != len(entries):
                raise ValueError("MinerU upload preparation returned incomplete batch metadata")

            with self._client() as client:
                for entry, upload_url in zip(entries, upload_urls, strict=False):
                    response = client.put(upload_url, content=entry["payload"])
                    response.raise_for_status()

            results = self._poll_batch_task(batch_id)
            result_by_id = {str(item.get("data_id")): item for item in results if item.get("data_id")}
            for entry in entries:
                result = result_by_id.get(entry["document_id"])
                if result is None:
                    raise ValueError(f"MinerU batch result missing for {entry['filename']}")
                zip_url = result.get("full_zip_url")
                if not zip_url:
                    raise ValueError(f"MinerU batch result for {entry['filename']} has no full_zip_url")
                documents.append(
                    PreparedDocument(
                        document_id=entry["document_id"],
                        title=entry["title"],
                        content=self._download_markdown(zip_url),
                        source_type="upload",
                        source_value=entry["filename"],
                        mime_type="text/markdown",
                        metadata={
                            "source_file_name": entry["filename"],
                            "source_mime_type": entry["mime_type"],
                            "mineru_batch_id": batch_id,
                            "mineru_backend": "api",
                            "mineru_model_version": model_version,
                        },
                    )
                )
        return documents


class AppGateway:
    def __init__(self, settings: AppSettings, catalog: CatalogStore):
        self.settings = settings
        self.catalog = catalog
        # This client only talks to the local Llama Stack server, so bypass shell proxy settings.
        self.client = LlamaStackClient(
            base_url=settings.llama_stack_base_url,
            http_client=httpx.Client(trust_env=False),
        )
        # The local stack server is from this repo (0.2.17) while the pip client may be newer.
        # Force the compatibility header to the server version so local app calls are accepted.
        self.client._custom_headers["X-LlamaStack-Client-Version"] = "0.2.17"
        self.mineru_client = MinerUClient(settings)
        self.openai_client = OpenAICompatibleClient(settings)
        self.chat_sessions: dict[str, ChatSessionState] = {}

    @staticmethod
    def _provider_supports_keyword_search(provider_id: str | None) -> bool:
        return provider_id in {"remote::pgvector", "pgvector", "remote::opengauss", "opengauss"}

    def _provider_id_for_vector_db(self, vector_db_id: str) -> str:
        match = next((kb.provider_id for kb in self._list_knowledge_bases() if kb.identifier == vector_db_id), None)
        return match or self.settings.default_provider_id

    def _normalize_mode_for_provider(self, provider_id: str, mode: str) -> str:
        if mode in {"keyword", "hybrid"} and not self._provider_supports_keyword_search(provider_id):
            return "vector"
        return mode

    @staticmethod
    def _list_resource_items(response: Any) -> list[Any]:
        data = getattr(response, "data", None)
        if data is not None:
            return list(data)
        return list(response)

    def _llama_headers(self) -> dict[str, str]:
        return {"X-LlamaStack-Client-Version": "0.2.17"}

    def _llama_request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        with httpx.Client(timeout=60.0, trust_env=False) as client:
            response = client.request(
                method,
                f"{self.settings.llama_stack_base_url.rstrip('/')}{path}",
                headers={**self._llama_headers(), **kwargs.pop("headers", {})},
                **kwargs,
            )
            response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}

    def _list_llama_models(self) -> list[ModelInfo]:
        models = []
        for model in self._list_resource_items(self.client.models.list()):
            models.append(
                ModelInfo(
                    identifier=getattr(model, "identifier", getattr(model, "id", "")),
                    model_type=getattr(model, "model_type", "unknown"),
                    embedding_dimension=(getattr(model, "metadata", {}) or {}).get("embedding_dimension"),
                )
            )
        return models

    def _list_embedding_models(self) -> list[ModelInfo]:
        return [model for model in self._list_llama_models() if model.model_type == "embedding"]

    def _default_embedding_model_id(self, models: list[ModelInfo]) -> str | None:
        preferred_ids = [
            "openai/Qwen/Qwen3-Embedding-0.6B",
            "Qwen/Qwen3-Embedding-0.6B",
        ]
        for preferred_id in preferred_ids:
            match = next((model.identifier for model in models if model.identifier == preferred_id), None)
            if match is not None:
                return match
        return next((model.identifier for model in models if model.model_type == "embedding"), None)

    def _embedding_dimension_for(self, embedding_model_id: str) -> int:
        model = next((item for item in self._list_embedding_models() if item.identifier == embedding_model_id), None)
        if model is None or model.embedding_dimension is None:
            raise ValueError(f"Embedding model {embedding_model_id} is unavailable or missing dimension metadata")
        return model.embedding_dimension

    def _list_knowledge_bases(self) -> list[KnowledgeBaseInfo]:
        payload = self._llama_request_json("GET", "/v1/vector-dbs")
        stores = payload.get("data") or []
        return [
            KnowledgeBaseInfo(
                identifier=store["identifier"],
                provider_id=store.get("provider_id") or self.settings.default_provider_id,
                embedding_model=store.get("embedding_model") or "",
                embedding_dimension=int(store.get("embedding_dimension") or 0),
                vector_db_name=store.get("vector_db_name"),
            )
            for store in stores
        ]

    def _query_chunks(
        self,
        vector_db_id: str,
        query: str,
        mode: str,
        max_chunks: int,
        ranker_type: str,
        alpha: float,
        impact_factor: float,
        score_threshold: float = 0.0,
    ) -> list[ChunkResult]:
        provider_id = self._provider_id_for_vector_db(vector_db_id)
        mode = self._normalize_mode_for_provider(provider_id, mode)
        params: dict[str, Any] = {
            "mode": mode,
            "max_chunks": max_chunks,
            "score_threshold": score_threshold,
        }
        if mode == "hybrid":
            if ranker_type == "weighted":
                params["ranker"] = {"type": "weighted", "alpha": alpha}
            else:
                params["ranker"] = {"type": "rrf", "impact_factor": impact_factor}

        response = self._llama_request_json(
            "POST",
            "/v1/vector-io/query",
            json={"vector_db_id": vector_db_id, "query": query, "params": params},
        )
        chunks: list[ChunkResult] = []
        for chunk, score in zip(response.get("chunks", []), response.get("scores", []), strict=False):
            metadata = dict(chunk.get("metadata", {}) or {})
            chunk_metadata = chunk.get("chunk_metadata")
            if isinstance(chunk_metadata, dict):
                document_id = chunk_metadata.get("document_id") or metadata.get("document_id")
            else:
                document_id = metadata.get("document_id")
            chunks.append(
                ChunkResult(
                    chunk_id=chunk.get("chunk_id"),
                    content=str(chunk.get("content", "")),
                    document_id=document_id,
                    score=float(score),
                    metadata=metadata,
                )
            )
        return chunks

    def _rewrite_query_if_enabled(self, model_id: str | None, query: str, enabled: bool) -> str:
        if not enabled:
            return query
        active_model_id = model_id or self.settings.openai_model or self.bootstrap().default_chat_model_id
        if not active_model_id:
            return query
        try:
            return self.openai_client.rewrite_query(active_model_id, query)
        except Exception:
            return query

    def _ingest_prepared_documents(
        self,
        vector_db_id: str,
        prepared_documents: list[PreparedDocument],
        chunking_strategy: ChunkingStrategy,
        chunk_size_in_tokens: int,
        chunk_size_in_chars: int,
    ) -> list[DocumentInfo]:
        chunks: list[dict[str, Any]] = []
        chunks_by_document: dict[str, list[DocumentChunkInfo]] = {}
        for document in prepared_documents:
            text_chunks, effective_chunking_strategy = _chunk_document_content(
                document,
                chunking_strategy,
                chunk_size_in_tokens,
                chunk_size_in_chars,
            )
            document_chunks: list[DocumentChunkInfo] = []
            for index, chunk_content in enumerate(text_chunks):
                chunk_metadata = {
                    "document_id": document.document_id,
                    "title": document.title,
                    "mime_type": document.mime_type,
                    "source_type": document.source_type,
                    "source_value": document.source_value,
                    "chunk_index": index,
                    "chunking_strategy": effective_chunking_strategy,
                    "requested_chunking_strategy": chunking_strategy,
                    **document.metadata,
                }
                chunk_id = f"{document.document_id}-chunk-{index}"
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": chunk_content,
                        "metadata": chunk_metadata,
                        "chunk_metadata": {
                            "document_id": document.document_id,
                            "source": document.source_value,
                        },
                    }
                )
                document_chunks.append(
                    DocumentChunkInfo(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        content=chunk_content,
                        metadata=chunk_metadata,
                    )
                )
            chunks_by_document[document.document_id] = document_chunks

        self._llama_request_json(
            "POST",
            "/v1/vector-io/insert",
            json={
                "vector_db_id": vector_db_id,
                "chunks": chunks,
            },
        )
        catalog_entries = [
            DocumentInfo(
                document_id=document.document_id,
                title=document.title,
                source_type=document.source_type,
                source_value=document.source_value,
                mime_type=document.mime_type,
                created_at=self.catalog.now_iso(),
                metadata=document.metadata,
            )
            for document in prepared_documents
        ]
        self.catalog.add_documents(vector_db_id, catalog_entries)
        for document_id, document_chunks in chunks_by_document.items():
            self.catalog.add_document_chunks(vector_db_id, document_id, document_chunks)
        return catalog_entries

    def _build_grounded_messages(
        self,
        state: ChatSessionState,
        user_message: str,
        chunks: list[ChunkResult],
    ) -> list[dict[str, str]]:
        context = "\n\n".join(
            [
                f"[{index}] document_id={chunk.document_id or 'unknown'} score={chunk.score:.4f}\n{chunk.content}"
                for index, chunk in enumerate(chunks, start=1)
            ]
        )
        system_message = (
            f"{state.instructions.strip()}\n\n"
            "Grounding rules:\n"
            "- Use the retrieved context as the primary source of truth.\n"
            "- If the context is insufficient, say that explicitly.\n"
            "- Do not invent citations or facts outside the provided context.\n"
            "- Keep the answer concise and directly useful.\n"
            "- Write a normal answer, not JSON, not a tool plan, and not chain-of-thought.\n"
            "- When you use a retrieved passage, cite it inline with [1], [2] style markers.\n"
        )
        prompt = (
            "Answer the user's latest question using the retrieved context.\n"
            "Prefer Chinese unless the user explicitly asks for another language.\n\n"
            f"Retrieved context:\n{context}\n\n"
            f"User question:\n{user_message}"
        )
        return [{"role": "system", "content": system_message}, *state.history, {"role": "user", "content": prompt}]

    def bootstrap(self) -> BootstrapResponse:
        embedding_models = self._list_embedding_models()
        default_embedding_model_id = self._default_embedding_model_id(embedding_models)
        chat_models, openai_configured = self.openai_client.list_models()
        default_chat_model_id = self.settings.openai_model or (chat_models[0] if chat_models else None)
        provider_available = any(
            getattr(provider, "api", None) == "vector_io"
            and getattr(provider, "provider_id", None) == self.settings.default_provider_id
            for provider in self._list_resource_items(self.client.providers.list())
        )
        return BootstrapResponse(
            llama_stack_base_url=self.settings.llama_stack_base_url,
            openai_base_url=self.settings.openai_base_url,
            mineru_base_url=self.settings.mineru_base_url,
            default_provider_id=self.settings.default_provider_id,
            default_query_rewrite=self.settings.default_query_rewrite,
            default_chat_model_id=default_chat_model_id,
            default_embedding_model_id=default_embedding_model_id,
            embedding_models=embedding_models,
            chat_models=chat_models,
            knowledge_bases=self._list_knowledge_bases(),
            provider_available=provider_available,
            openai_configured=openai_configured,
            mineru_configured=bool(self.settings.mineru_api_token),
        )

    def create_knowledge_base(self, request: CreateKnowledgeBaseRequest) -> KnowledgeBaseInfo:
        provider_id = request.provider_id or self.settings.default_provider_id
        embedding_model = request.embedding_model or self.bootstrap().default_embedding_model_id
        if embedding_model is None:
            raise ValueError("No embedding model is available in the connected Llama Stack server")

        embedding_dimension = request.embedding_dimension or self._embedding_dimension_for(embedding_model)

        vector_db_id = request.vector_db_id or f"{_slugify(request.name)}-{uuid4().hex[:8]}"
        vector_db = self._llama_request_json(
            "POST",
            "/v1/vector-dbs",
            json={
                "vector_db_id": vector_db_id,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "provider_id": provider_id,
                "vector_db_name": request.name,
                "provider_vector_db_id": vector_db_id,
            },
        )
        return KnowledgeBaseInfo(
            identifier=vector_db.get("identifier", vector_db_id),
            provider_id=provider_id,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            vector_db_name=request.name,
        )

    def delete_knowledge_base(self, vector_db_id: str) -> None:
        try:
            self._llama_request_json("DELETE", f"/v1/vector-dbs/{vector_db_id}")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
        self.catalog.remove_knowledge_base(vector_db_id)
        stale_sessions = [session_id for session_id, state in self.chat_sessions.items() if state.config.vector_db_id == vector_db_id]
        for session_id in stale_sessions:
            self.chat_sessions.pop(session_id, None)

    def _delete_pgvector_document_rows(self, vector_db_id: str, document_id: str, chunk_ids: list[str]) -> None:
        psycopg2 = _load_psycopg2()
        table_name = f"vector_store_{_make_sql_identifier(vector_db_id)}"
        chunk_ids = chunk_ids or [f"{document_id}__missing_chunk_snapshot__"]
        conn = psycopg2.connect(
            host=self.settings.pgvector_host,
            port=self.settings.pgvector_port,
            database=self.settings.pgvector_db,
            user=self.settings.pgvector_user,
            password=self.settings.pgvector_password,
            connect_timeout=self.settings.pgvector_connect_timeout,
        )
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {table_name}
                    WHERE id = ANY(%s)
                       OR document->'chunk_metadata'->>'document_id' = %s
                       OR document->'metadata'->>'document_id' = %s
                    """,
                    (chunk_ids, document_id, document_id),
                )
        finally:
            conn.close()

    def delete_document(self, vector_db_id: str, document_id: str) -> None:
        provider_id = self._provider_id_for_vector_db(vector_db_id)
        if provider_id != "remote::pgvector":
            raise ValueError(f"Document deletion is only supported for pgvector-backed knowledge bases, got: {provider_id}")

        removed_document = next(
            (document for document in self.catalog.list_documents(vector_db_id) if document.document_id == document_id),
            None,
        )
        removed_chunks = self.catalog.list_document_chunks(vector_db_id, document_id)
        if removed_document is None and not removed_chunks:
            raise ValueError(f"Unknown document: {document_id}")

        chunk_ids = [chunk.chunk_id for chunk in removed_chunks]
        self._delete_pgvector_document_rows(vector_db_id, document_id, chunk_ids)
        self.catalog.remove_document(vector_db_id, document_id)

        for session_id, state in list(self.chat_sessions.items()):
            if state.config.vector_db_id != vector_db_id:
                continue
            state.history = [
                entry for entry in state.history if document_id not in entry.get("content", "")
            ]

    def list_documents(self, vector_db_id: str) -> list[DocumentInfo]:
        return self.catalog.list_documents(vector_db_id)

    def list_document_chunks(self, vector_db_id: str, document_id: str) -> list[DocumentChunkInfo]:
        return self.catalog.list_document_chunks(vector_db_id, document_id)

    def ingest_text_documents(
        self,
        vector_db_id: str,
        inputs: list[TextDocumentInput],
        chunking_strategy: ChunkingStrategy,
        chunk_size_in_tokens: int,
        chunk_size_in_chars: int,
    ) -> list[DocumentInfo]:
        chunking_strategy = _normalize_chunking_strategy(chunking_strategy)
        prepared_documents = [
            PreparedDocument(
                document_id=item.document_id or f"doc_{uuid4().hex[:10]}",
                title=item.title,
                content=item.text,
                source_type="text",
                source_value=_text_preview(item.text, 120),
                mime_type="text/plain",
                metadata=item.metadata,
            )
            for item in inputs
        ]
        return self._ingest_prepared_documents(
            vector_db_id,
            prepared_documents,
            chunking_strategy,
            chunk_size_in_tokens,
            chunk_size_in_chars,
        )

    def ingest_urls(
        self,
        vector_db_id: str,
        urls: list[str],
        chunking_strategy: ChunkingStrategy,
        chunk_size_in_tokens: int,
        chunk_size_in_chars: int,
    ) -> list[DocumentInfo]:
        chunking_strategy = _normalize_chunking_strategy(chunking_strategy)
        prepared_documents = self.mineru_client.extract_urls(urls)
        return self._ingest_prepared_documents(
            vector_db_id,
            prepared_documents,
            chunking_strategy,
            chunk_size_in_tokens,
            chunk_size_in_chars,
        )

    def ingest_uploads(
        self,
        vector_db_id: str,
        files: list[UploadFile],
        chunking_strategy: ChunkingStrategy,
        chunk_size_in_tokens: int,
        chunk_size_in_chars: int,
    ) -> list[DocumentInfo]:
        chunking_strategy = _normalize_chunking_strategy(chunking_strategy)
        prepared_documents = self.mineru_client.extract_uploads(files)
        return self._ingest_prepared_documents(
            vector_db_id,
            prepared_documents,
            chunking_strategy,
            chunk_size_in_tokens,
            chunk_size_in_chars,
        )

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        rewritten_query = self._rewrite_query_if_enabled(
            self.settings.openai_model or self.bootstrap().default_chat_model_id,
            request.query,
            request.query_rewrite,
        )
        chunks = self._query_chunks(
            vector_db_id=request.vector_db_id,
            query=rewritten_query,
            mode=request.mode,
            max_chunks=request.max_chunks,
            ranker_type=request.ranker_type,
            alpha=request.alpha,
            impact_factor=request.impact_factor,
            score_threshold=request.score_threshold,
        )
        provider_id = self._provider_id_for_vector_db(request.vector_db_id)
        effective_mode = self._normalize_mode_for_provider(provider_id, request.mode)
        return RetrievalResponse(mode=effective_mode, vector_db_id=request.vector_db_id, chunks=chunks)

    def create_chat_session(self, request: CreateChatSessionRequest) -> ChatSessionInfo:
        bootstrap = self.bootstrap()
        model_id = request.model_id or bootstrap.default_chat_model_id
        if model_id is None:
            raise ValueError("No OpenAI-compatible chat model is configured")

        session_id = f"chat_{uuid4().hex[:12]}"
        session_info = ChatSessionInfo(
            session_id=session_id,
            vector_db_id=request.vector_db_id,
            model_id=model_id,
            mode=self._normalize_mode_for_provider(self._provider_id_for_vector_db(request.vector_db_id), request.mode),
            query_rewrite=request.query_rewrite,
            max_chunks=request.max_chunks,
            ranker_type=request.ranker_type,
        )
        self.chat_sessions[session_id] = ChatSessionState(
            config=session_info,
            instructions=request.instructions,
            alpha=request.alpha,
            impact_factor=request.impact_factor,
        )
        return session_info

    def send_chat_message(self, session_id: str, message: str) -> ChatMessageResponse:
        state = self.chat_sessions.get(session_id)
        if state is None:
            raise ValueError(f"Unknown chat session: {session_id}")

        chunks = self._query_chunks(
            vector_db_id=state.config.vector_db_id,
            query=self._rewrite_query_if_enabled(state.config.model_id, message, state.config.query_rewrite),
            mode=state.config.mode,
            max_chunks=state.config.max_chunks,
            ranker_type=state.config.ranker_type,
            alpha=state.alpha,
            impact_factor=state.impact_factor,
        )
        if not chunks:
            answer = "I could not find grounded context for this question in the selected knowledge base."
            citations: list[CitationInfo] = []
        else:
            messages = self._build_grounded_messages(state, message, chunks)
            answer = self.openai_client.chat_completion(state.config.model_id, messages).strip()
            cited_indices = _extract_citation_indices(answer, len(chunks))
            if not cited_indices:
                cited_indices = list(range(1, min(len(chunks), 2) + 1))
            citations = [
                CitationInfo(
                    citation_index=index,
                    document_id=chunks[index - 1].document_id,
                    score=chunks[index - 1].score,
                    snippet=_text_preview(chunks[index - 1].content, 420),
                )
                for index in cited_indices
            ]

        state.history.append({"role": "user", "content": message})
        state.history.append({"role": "assistant", "content": answer})
        return ChatMessageResponse(session_id=session_id, answer=answer, citations=citations)
