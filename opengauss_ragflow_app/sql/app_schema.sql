CREATE SCHEMA IF NOT EXISTS rag_app;
SET search_path TO rag_app, public;

CREATE OR REPLACE FUNCTION rag_app.touch_updated_at()
RETURNS TRIGGER
AS $$
BEGIN
    NEW.updated_at := CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS rag_app.knowledge_base (
    knowledge_base_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    vector_db_id VARCHAR(128) NOT NULL UNIQUE,
    provider_id VARCHAR(128) NOT NULL,
    embedding_model VARCHAR(255) NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ck_knowledge_base_embedding_dimension CHECK (embedding_dimension > 0)
);

CREATE TABLE IF NOT EXISTS rag_app.document (
    document_id VARCHAR(64) PRIMARY KEY,
    knowledge_base_id VARCHAR(64) NOT NULL,
    title VARCHAR(255) NOT NULL,
    source_type VARCHAR(16) NOT NULL,
    source_value TEXT NOT NULL,
    mime_type VARCHAR(255),
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    content_markdown TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_document_knowledge_base
        FOREIGN KEY (knowledge_base_id) REFERENCES rag_app.knowledge_base (knowledge_base_id) ON DELETE CASCADE,
    CONSTRAINT ck_document_source_type
        CHECK (source_type IN ('text', 'url', 'upload')),
    CONSTRAINT ck_document_status
        CHECK (status IN ('pending', 'extracting', 'extracted', 'chunked', 'indexed', 'failed'))
);

CREATE TABLE IF NOT EXISTS rag_app.document_asset (
    asset_id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL,
    asset_type VARCHAR(24) NOT NULL,
    file_name VARCHAR(255),
    locator TEXT,
    mime_type VARCHAR(255),
    size_bytes BIGINT,
    checksum VARCHAR(128),
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_document_asset_document
        FOREIGN KEY (document_id) REFERENCES rag_app.document (document_id) ON DELETE CASCADE,
    CONSTRAINT ck_document_asset_type
        CHECK (asset_type IN ('original_file', 'source_url', 'extracted_zip', 'extracted_markdown', 'derived_text')),
    CONSTRAINT ck_document_asset_size
        CHECK (size_bytes IS NULL OR size_bytes >= 0)
);

CREATE TABLE IF NOT EXISTS rag_app.mineru_task (
    task_id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL,
    task_type VARCHAR(24) NOT NULL,
    mineru_task_id VARCHAR(128),
    mineru_batch_id VARCHAR(128),
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(16) NOT NULL DEFAULT 'pending',
    result_zip_url TEXT,
    result_markdown_locator TEXT,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    finished_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_mineru_task_document
        FOREIGN KEY (document_id) REFERENCES rag_app.document (document_id) ON DELETE CASCADE,
    CONSTRAINT ck_mineru_task_type
        CHECK (task_type IN ('url_extract', 'file_extract')),
    CONSTRAINT ck_mineru_task_status
        CHECK (status IN ('pending', 'submitted', 'running', 'succeeded', 'failed', 'cancelled'))
);

CREATE TABLE IF NOT EXISTS rag_app.chunk_record (
    chunk_record_id BIGSERIAL PRIMARY KEY,
    knowledge_base_id VARCHAR(64) NOT NULL,
    document_id VARCHAR(64) NOT NULL,
    vector_db_id VARCHAR(128) NOT NULL,
    chunk_id VARCHAR(128) NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_chunk_record_knowledge_base
        FOREIGN KEY (knowledge_base_id) REFERENCES rag_app.knowledge_base (knowledge_base_id) ON DELETE CASCADE,
    CONSTRAINT fk_chunk_record_document
        FOREIGN KEY (document_id) REFERENCES rag_app.document (document_id) ON DELETE CASCADE,
    CONSTRAINT uq_chunk_record_vector_db_chunk UNIQUE (vector_db_id, chunk_id),
    CONSTRAINT uq_chunk_record_document_index UNIQUE (document_id, chunk_index),
    CONSTRAINT ck_chunk_record_chunk_index CHECK (chunk_index >= 0),
    CONSTRAINT ck_chunk_record_token_count CHECK (token_count IS NULL OR token_count >= 0)
);

CREATE TABLE IF NOT EXISTS rag_app.chat_session (
    session_id VARCHAR(64) PRIMARY KEY,
    knowledge_base_id VARCHAR(64) NOT NULL,
    vector_db_id VARCHAR(128) NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    instructions TEXT NOT NULL,
    mode VARCHAR(16) NOT NULL,
    max_chunks INTEGER NOT NULL,
    ranker_type VARCHAR(16) NOT NULL,
    alpha NUMERIC(6, 4),
    impact_factor NUMERIC(10, 4),
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_chat_session_knowledge_base
        FOREIGN KEY (knowledge_base_id) REFERENCES rag_app.knowledge_base (knowledge_base_id) ON DELETE CASCADE,
    CONSTRAINT ck_chat_session_mode
        CHECK (mode IN ('vector', 'keyword', 'hybrid')),
    CONSTRAINT ck_chat_session_ranker_type
        CHECK (ranker_type IN ('rrf', 'weighted')),
    CONSTRAINT ck_chat_session_status
        CHECK (status IN ('active', 'closed', 'failed')),
    CONSTRAINT ck_chat_session_max_chunks
        CHECK (max_chunks > 0),
    CONSTRAINT ck_chat_session_alpha
        CHECK (alpha IS NULL OR (alpha >= 0 AND alpha <= 1)),
    CONSTRAINT ck_chat_session_impact_factor
        CHECK (impact_factor IS NULL OR impact_factor > 0)
);

CREATE TABLE IF NOT EXISTS rag_app.chat_message (
    chat_message_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    role VARCHAR(16) NOT NULL,
    content TEXT NOT NULL,
    citations_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_chat_message_session
        FOREIGN KEY (session_id) REFERENCES rag_app.chat_session (session_id) ON DELETE CASCADE,
    CONSTRAINT ck_chat_message_role
        CHECK (role IN ('system', 'user', 'assistant')),
    CONSTRAINT ck_chat_message_prompt_tokens
        CHECK (prompt_tokens IS NULL OR prompt_tokens >= 0),
    CONSTRAINT ck_chat_message_completion_tokens
        CHECK (completion_tokens IS NULL OR completion_tokens >= 0),
    CONSTRAINT ck_chat_message_total_tokens
        CHECK (total_tokens IS NULL OR total_tokens >= 0)
);

CREATE TABLE IF NOT EXISTS rag_app.retrieval_log (
    retrieval_log_id BIGSERIAL PRIMARY KEY,
    knowledge_base_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64),
    vector_db_id VARCHAR(128) NOT NULL,
    query_text TEXT NOT NULL,
    mode VARCHAR(16) NOT NULL,
    ranker_type VARCHAR(16),
    alpha NUMERIC(6, 4),
    impact_factor NUMERIC(10, 4),
    max_chunks INTEGER NOT NULL,
    score_threshold NUMERIC(10, 6),
    result_count INTEGER NOT NULL DEFAULT 0,
    latency_ms INTEGER,
    results_json JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_retrieval_log_knowledge_base
        FOREIGN KEY (knowledge_base_id) REFERENCES rag_app.knowledge_base (knowledge_base_id) ON DELETE CASCADE,
    CONSTRAINT fk_retrieval_log_session
        FOREIGN KEY (session_id) REFERENCES rag_app.chat_session (session_id) ON DELETE SET NULL,
    CONSTRAINT ck_retrieval_log_mode
        CHECK (mode IN ('vector', 'keyword', 'hybrid')),
    CONSTRAINT ck_retrieval_log_ranker_type
        CHECK (ranker_type IS NULL OR ranker_type IN ('rrf', 'weighted')),
    CONSTRAINT ck_retrieval_log_alpha
        CHECK (alpha IS NULL OR (alpha >= 0 AND alpha <= 1)),
    CONSTRAINT ck_retrieval_log_impact_factor
        CHECK (impact_factor IS NULL OR impact_factor > 0),
    CONSTRAINT ck_retrieval_log_max_chunks
        CHECK (max_chunks > 0),
    CONSTRAINT ck_retrieval_log_result_count
        CHECK (result_count >= 0),
    CONSTRAINT ck_retrieval_log_latency_ms
        CHECK (latency_ms IS NULL OR latency_ms >= 0)
);

CREATE TABLE IF NOT EXISTS rag_app.model_call_log (
    model_call_log_id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64),
    chat_message_id BIGINT,
    provider_name VARCHAR(64) NOT NULL DEFAULT 'openai-compatible',
    model_id VARCHAR(255) NOT NULL,
    request_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status VARCHAR(16) NOT NULL DEFAULT 'succeeded',
    latency_ms INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_model_call_log_session
        FOREIGN KEY (session_id) REFERENCES rag_app.chat_session (session_id) ON DELETE SET NULL,
    CONSTRAINT fk_model_call_log_chat_message
        FOREIGN KEY (chat_message_id) REFERENCES rag_app.chat_message (chat_message_id) ON DELETE SET NULL,
    CONSTRAINT ck_model_call_log_status
        CHECK (status IN ('submitted', 'succeeded', 'failed')),
    CONSTRAINT ck_model_call_log_latency_ms
        CHECK (latency_ms IS NULL OR latency_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_document_knowledge_base_id
    ON rag_app.document (knowledge_base_id);

CREATE INDEX IF NOT EXISTS idx_document_status
    ON rag_app.document (status);

CREATE INDEX IF NOT EXISTS idx_document_asset_document_id
    ON rag_app.document_asset (document_id);

CREATE INDEX IF NOT EXISTS idx_mineru_task_document_id
    ON rag_app.mineru_task (document_id);

CREATE INDEX IF NOT EXISTS idx_mineru_task_status
    ON rag_app.mineru_task (status);

CREATE INDEX IF NOT EXISTS idx_mineru_task_remote_ids
    ON rag_app.mineru_task (mineru_task_id, mineru_batch_id);

CREATE INDEX IF NOT EXISTS idx_chunk_record_document_id
    ON rag_app.chunk_record (document_id);

CREATE INDEX IF NOT EXISTS idx_chunk_record_knowledge_base_id
    ON rag_app.chunk_record (knowledge_base_id);

CREATE INDEX IF NOT EXISTS idx_chunk_record_vector_db_id
    ON rag_app.chunk_record (vector_db_id);

CREATE INDEX IF NOT EXISTS idx_chat_session_knowledge_base_id
    ON rag_app.chat_session (knowledge_base_id);

CREATE INDEX IF NOT EXISTS idx_chat_session_status
    ON rag_app.chat_session (status);

CREATE INDEX IF NOT EXISTS idx_chat_message_session_id
    ON rag_app.chat_message (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_log_knowledge_base_id
    ON rag_app.retrieval_log (knowledge_base_id, created_at);

CREATE INDEX IF NOT EXISTS idx_retrieval_log_session_id
    ON rag_app.retrieval_log (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_model_call_log_session_id
    ON rag_app.model_call_log (session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_model_call_log_chat_message_id
    ON rag_app.model_call_log (chat_message_id);

DROP TRIGGER IF EXISTS trg_knowledge_base_updated_at ON rag_app.knowledge_base;
CREATE TRIGGER trg_knowledge_base_updated_at
BEFORE UPDATE ON rag_app.knowledge_base
FOR EACH ROW
EXECUTE PROCEDURE rag_app.touch_updated_at();

DROP TRIGGER IF EXISTS trg_document_updated_at ON rag_app.document;
CREATE TRIGGER trg_document_updated_at
BEFORE UPDATE ON rag_app.document
FOR EACH ROW
EXECUTE PROCEDURE rag_app.touch_updated_at();

DROP TRIGGER IF EXISTS trg_mineru_task_updated_at ON rag_app.mineru_task;
CREATE TRIGGER trg_mineru_task_updated_at
BEFORE UPDATE ON rag_app.mineru_task
FOR EACH ROW
EXECUTE PROCEDURE rag_app.touch_updated_at();

DROP TRIGGER IF EXISTS trg_chat_session_updated_at ON rag_app.chat_session;
CREATE TRIGGER trg_chat_session_updated_at
BEFORE UPDATE ON rag_app.chat_session
FOR EACH ROW
EXECUTE PROCEDURE rag_app.touch_updated_at();
