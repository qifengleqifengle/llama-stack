from typing import Any, Literal

from pydantic import BaseModel, Field


SearchMode = Literal["vector", "keyword", "hybrid"]
RankerType = Literal["rrf", "weighted"]


class ModelInfo(BaseModel):
    identifier: str
    model_type: str
    embedding_dimension: int | None = None


class KnowledgeBaseInfo(BaseModel):
    identifier: str
    provider_id: str
    embedding_model: str
    embedding_dimension: int
    vector_db_name: str | None = None


class DocumentInfo(BaseModel):
    document_id: str
    title: str
    source_type: Literal["text", "url", "upload"]
    source_value: str
    mime_type: str | None = None
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunkInfo(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BootstrapResponse(BaseModel):
    llama_stack_base_url: str
    openai_base_url: str
    mineru_base_url: str
    default_provider_id: str
    default_chat_model_id: str | None = None
    default_embedding_model_id: str | None = None
    embedding_models: list[ModelInfo]
    chat_models: list[str]
    knowledge_bases: list[KnowledgeBaseInfo]
    provider_available: bool
    openai_configured: bool
    mineru_configured: bool


class CreateKnowledgeBaseRequest(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    vector_db_id: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    provider_id: str | None = None


class TextDocumentInput(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1)
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TextIngestRequest(BaseModel):
    documents: list[TextDocumentInput]
    chunk_size_in_tokens: int = Field(default=512, ge=64, le=4096)


class UrlIngestRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    chunk_size_in_tokens: int = Field(default=512, ge=64, le=4096)


class RetrievalRequest(BaseModel):
    vector_db_id: str
    query: str = Field(min_length=1)
    mode: SearchMode = "hybrid"
    max_chunks: int = Field(default=6, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0)
    ranker_type: RankerType = "rrf"
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    impact_factor: float = Field(default=60.0, gt=0.0)


class ChunkResult(BaseModel):
    chunk_id: str | None = None
    content: str
    document_id: str | None = None
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    mode: SearchMode
    vector_db_id: str
    chunks: list[ChunkResult]


class CreateChatSessionRequest(BaseModel):
    vector_db_id: str
    model_id: str | None = None
    instructions: str = Field(
        default="You are a grounded assistant. Use the retrieved knowledge base results and cite them faithfully."
    )
    mode: SearchMode = "hybrid"
    max_chunks: int = Field(default=5, ge=1, le=20)
    ranker_type: RankerType = "rrf"
    alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    impact_factor: float = Field(default=60.0, gt=0.0)


class ChatSessionInfo(BaseModel):
    session_id: str
    vector_db_id: str
    model_id: str
    mode: SearchMode
    max_chunks: int
    ranker_type: RankerType


class ChatMessageRequest(BaseModel):
    message: str = Field(min_length=1)


class CitationInfo(BaseModel):
    citation_index: int | None = None
    document_id: str | None = None
    score: float | None = None
    snippet: str


class ChatMessageResponse(BaseModel):
    session_id: str
    answer: str
    citations: list[CitationInfo]
