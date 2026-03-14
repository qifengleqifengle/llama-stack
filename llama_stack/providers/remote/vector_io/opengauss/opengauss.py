# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import re
from contextlib import contextmanager
from typing import Any

import psycopg2
from numpy.typing import NDArray
from psycopg2 import sql
from psycopg2.extras import Json, execute_values
from pydantic import BaseModel, TypeAdapter

from llama_stack.apis.common.errors import VectorStoreNotFoundError
from llama_stack.apis.files.files import Files
from llama_stack.apis.inference import InterleavedContent
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    RERANKER_TYPE_WEIGHTED,
    ChunkForDeletion,
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import OpenGaussVectorIOConfig

log = logging.getLogger(__name__)

VERSION = "v6"
VECTOR_DBS_PREFIX = f"vector_dbs:opengauss:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:opengauss:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:opengauss:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:opengauss:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:opengauss:{VERSION}::"


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}

    min_score = min(scores.values())
    max_score = max(scores.values())
    score_range = max_score - min_score
    if score_range == 0:
        return dict.fromkeys(scores, 1.0)

    return {doc_id: (score - min_score) / score_range for doc_id, score in scores.items()}


def _weighted_rerank(
    vector_scores: dict[str, float],
    keyword_scores: dict[str, float],
    alpha: float = 0.5,
) -> dict[str, float]:
    all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
    normalized_vector_scores = _normalize_scores(vector_scores)
    normalized_keyword_scores = _normalize_scores(keyword_scores)

    return {
        doc_id: (alpha * normalized_vector_scores.get(doc_id, 0.0))
        + ((1 - alpha) * normalized_keyword_scores.get(doc_id, 0.0))
        for doc_id in all_ids
    }


def _rrf_rerank(
    vector_scores: dict[str, float],
    keyword_scores: dict[str, float],
    impact_factor: float = 60.0,
) -> dict[str, float]:
    vector_ranks = {
        doc_id: i + 1 for i, (doc_id, _) in enumerate(sorted(vector_scores.items(), key=lambda x: x[1], reverse=True))
    }
    keyword_ranks = {
        doc_id: i + 1
        for i, (doc_id, _) in enumerate(sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True))
    }
    all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())

    return {
        doc_id: (1.0 / (impact_factor + vector_ranks.get(doc_id, float("inf"))))
        + (1.0 / (impact_factor + keyword_ranks.get(doc_id, float("inf"))))
        for doc_id in all_ids
    }


def _make_sql_identifier(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _make_index_name(table_name: str, suffix: str) -> str:
    max_len = 63
    base = f"{table_name}_{suffix}"
    return base if len(base) <= max_len else f"{base[: max_len - 8]}_{abs(hash(base)) % 10**7:07d}"


def _make_regconfig_literal(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.]+", name):
        raise ValueError(f"Unsupported OpenGauss FTS config: {name}")
    return f"'{name}'"


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(format(float(v), ".17g") for v in values) + "]"


def upsert_models(conn, keys_models: list[tuple[str, BaseModel]]):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = sql.SQL(
            """
            MERGE INTO metadata_store AS target
            USING (VALUES %s) AS source (key, data)
            ON (target.key = source.key)
            WHEN MATCHED THEN
                UPDATE SET data = source.data
            WHEN NOT MATCHED THEN
                INSERT (key, data) VALUES (source.key, source.data);
            """
        )

        values = [(key, Json(model.model_dump())) for key, model in keys_models]
        execute_values(cur, query, values, template="(%s, %s::JSONB)")


def load_models(cur, cls):
    cur.execute("SELECT key, data FROM metadata_store")
    rows = cur.fetchall()
    return [TypeAdapter(cls).validate_python(row["data"]) for row in rows]


def connect_opengauss(config: OpenGaussVectorIOConfig):
    conn = psycopg2.connect(
        host=config.host,
        port=config.port,
        database=config.db,
        user=config.user,
        password=config.password,
        connect_timeout=config.connect_timeout,
    )
    conn.autocommit = True
    return conn


def _get_existing_vector_dimension(cur, table_name: str) -> int | None:
    cur.execute(
        """
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        JOIN pg_class c ON a.attrelid = c.oid
        JOIN pg_namespace n ON c.relnamespace = n.oid
        WHERE n.nspname = current_schema()
          AND c.relname = %s
          AND a.attname = 'embedding'
          AND a.attnum > 0
          AND NOT a.attisdropped
        """,
        (table_name,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None

    match = re.fullmatch(r"vector\((\d+)\)", row[0])
    if not match:
        raise RuntimeError(f"Unexpected embedding column type for {table_name}: {row[0]}")
    return int(match.group(1))


class OpenGaussIndex(EmbeddingIndex):
    def __init__(
        self,
        vector_db: VectorDB,
        dimension: int,
        config: OpenGaussVectorIOConfig,
        kvstore: KVStore | None = None,
    ):
        self.config = config
        self.use_hnsw = dimension <= 2000
        with self._cursor() as cur:
            sanitized_identifier = _make_sql_identifier(vector_db.identifier)
            self.vector_table_name = f"vector_store_{sanitized_identifier}"
            self.table_name = self.vector_table_name
            self.hnsw_index_name = _make_index_name(self.vector_table_name, "embedding_hnsw_idx")
            self.fts_index_name = _make_index_name(self.vector_table_name, "content_fts_gin_idx")
            self.fts_config_literal = _make_regconfig_literal(self.config.fts_config)
            self.kvstore = kvstore

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.vector_table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    document JSONB,
                    embedding vector({dimension})
                )
                """
            )
            existing_dimension = _get_existing_vector_dimension(cur, self.vector_table_name)
            if existing_dimension is not None and existing_dimension != dimension:
                raise RuntimeError(
                    f"Existing table {self.vector_table_name} uses vector({existing_dimension}), "
                    f"but vector DB {vector_db.identifier} expects vector({dimension}). "
                    "Drop the table or register a new vector_db identifier."
                )
            if self.use_hnsw:
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.hnsw_index_name}
                    ON {self.vector_table_name}
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = {self.config.hnsw_m}, ef_construction = {self.config.hnsw_ef_construction})
                    """
                )
            else:
                log.warning(
                    "OpenGauss HNSW disabled for vector_db=%s because embedding dimension %s exceeds the index limit",
                    vector_db.identifier,
                    dimension,
                )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.fts_index_name}
                ON {self.vector_table_name}
                USING gin (to_tsvector({self.fts_config_literal}, coalesce(content, '')))
                """
            )

    @contextmanager
    def _cursor(self):
        conn = connect_opengauss(self.config)
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                yield cur
        finally:
            conn.close()

    def _apply_vector_query_settings(self, cur) -> None:
        if self.use_hnsw:
            cur.execute(f"SET hnsw_ef_search = {self.config.hnsw_ef_search}")

    def _apply_keyword_query_settings(self, cur, k: int) -> None:
        return None

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )
        assert all(isinstance(chunk.content, str) for chunk in chunks), "OpenGauss keyword search only supports text chunks"

        vector_values = []
        for i, chunk in enumerate(chunks):
            vector_values.append(
                (
                    f"{chunk.chunk_id}",
                    chunk.content,
                    Json(chunk.model_dump()),
                    _vector_literal(embeddings[i].tolist()),
                )
            )

        with self._cursor() as cur:
            chunk_ids = [value[0] for value in vector_values]
            cur.execute(f"DELETE FROM {self.vector_table_name} WHERE id = ANY(%s)", (chunk_ids,))
            for chunk_id, content, document, embedding_literal in vector_values:
                cur.execute(
                    f"""
                    INSERT INTO {self.vector_table_name} (id, content, document, embedding)
                    VALUES (%s, %s, %s::JSONB, %s::VECTOR)
                    """,
                    (chunk_id, content, document, embedding_literal),
                )

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        with self._cursor() as cur:
            self._apply_vector_query_settings(cur)
            cur.execute(
                f"""
            SELECT document, embedding <=> %s::VECTOR AS distance
            FROM {self.vector_table_name}
            ORDER BY distance
            LIMIT %s
        """,
                (_vector_literal(embedding.tolist()), k),
            )
            results = cur.fetchall()

            chunks = []
            scores = []
            for doc, dist in results:
                score = 1.0 / float(dist) if dist != 0 else float("inf")
                if score < score_threshold:
                    continue
                chunks.append(Chunk(**doc))
                scores.append(score)

            return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_keyword(
        self,
        query_string: str,
        k: int,
        score_threshold: float,
    ) -> QueryChunksResponse:
        if not query_string.strip():
            return QueryChunksResponse(chunks=[], scores=[])

        with self._cursor() as cur:
            self._apply_keyword_query_settings(cur, k)
            cur.execute(
                f"""
                SELECT document,
                       ts_rank(
                           to_tsvector({self.fts_config_literal}, coalesce(content, '')),
                           plainto_tsquery({self.fts_config_literal}, %s)
                       ) AS score
                FROM {self.vector_table_name}
                WHERE to_tsvector({self.fts_config_literal}, coalesce(content, ''))
                      @@ plainto_tsquery({self.fts_config_literal}, %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_string, query_string, k),
            )
            results = cur.fetchall()

        chunks = []
        scores = []
        for row in results:
            score = row["score"]
            if score is None:
                continue
            score = float(score)
            if score < score_threshold:
                continue
            chunks.append(Chunk(**row["document"]))
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def query_hybrid(
        self,
        embedding: NDArray,
        query_string: str,
        k: int,
        score_threshold: float,
        reranker_type: str,
        reranker_params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        reranker_params = reranker_params or {}

        vector_response = await self.query_vector(embedding, k, score_threshold)
        keyword_response = await self.query_keyword(query_string, k, score_threshold)

        vector_scores = {
            chunk.chunk_id: score for chunk, score in zip(vector_response.chunks, vector_response.scores, strict=False)
        }
        keyword_scores = {
            chunk.chunk_id: score
            for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
        }

        if reranker_type == RERANKER_TYPE_WEIGHTED:
            combined_scores = _weighted_rerank(vector_scores, keyword_scores, reranker_params.get("alpha", 0.5))
        else:
            combined_scores = _rrf_rerank(
                vector_scores,
                keyword_scores,
                reranker_params.get("impact_factor", 60.0),
            )

        chunk_map = {chunk.chunk_id: chunk for chunk in vector_response.chunks + keyword_response.chunks}
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        chunks = []
        scores = []
        for chunk_id, score in sorted_items[:k]:
            if score < score_threshold or chunk_id not in chunk_map:
                continue
            chunks.append(chunk_map[chunk_id])
            scores.append(score)

        return QueryChunksResponse(chunks=chunks, scores=scores)

    async def delete(self):
        with self._cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.vector_table_name}")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Remove chunks from the OpenGauss vector and keyword tables."""
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        with self._cursor() as cur:
            cur.execute(f"DELETE FROM {self.vector_table_name} WHERE id = ANY(%s)", (chunk_ids,))


class OpenGaussVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: OpenGaussVectorIOConfig,
        inference_api: Any,
        files_api: Files | None = None,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.cache: dict[str, VectorDBWithIndex] = {}
        self.files_api = files_api
        self.kvstore: KVStore | None = None
        self.vector_db_store = None
        self.openai_vector_stores: dict[str, dict[str, Any]] = {}
        self.metadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        log.info(f"Initializing OpenGauss memory adapter with config: {self.config}")
        if self.config.kvstore is None:
            raise ValueError("OpenGauss vector IO requires a kvstore configuration")

        self.kvstore = await kvstore_impl(self.config.kvstore)
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_dbs = [VectorDB.model_validate_json(raw) for raw in await self.kvstore.values_in_range(start_key, end_key)]
        await self.initialize_openai_vector_stores()

        try:
            conn = connect_opengauss(self.config)
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()[0]
                    log.info(f"OpenGauss server version: {version}")
                    log.info("Assuming native vector support is enabled in this OpenGauss instance.")

                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS metadata_store (
                            key TEXT PRIMARY KEY,
                            data JSONB
                        )
                    """
                    )
            finally:
                conn.close()

            for vector_db in stored_vector_dbs:
                self.cache[vector_db.identifier] = VectorDBWithIndex(
                    vector_db=vector_db,
                    index=OpenGaussIndex(
                        vector_db,
                        vector_db.embedding_dimension,
                        self.config,
                        kvstore=self.kvstore,
                    ),
                    inference_api=self.inference_api,
                )
        except Exception as e:
            log.exception("Could not connect to OpenGauss database server")
            raise RuntimeError("Could not connect to OpenGauss database server") from e

    async def shutdown(self) -> None:
        log.info("OpenGauss vector IO adapter shutdown complete")

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        assert self.kvstore is not None
        index = VectorDBWithIndex(
            vector_db,
            index=OpenGaussIndex(
                vector_db,
                vector_db.embedding_dimension,
                self.config,
                kvstore=self.kvstore,
            ),
            inference_api=self.inference_api,
        )

        try:
            conn = connect_opengauss(self.config)
            try:
                upsert_models(conn, [(vector_db.identifier, vector_db)])
            finally:
                conn.close()
            await self.kvstore.set(key=f"{VECTOR_DBS_PREFIX}{vector_db.identifier}", value=vector_db.model_dump_json())
        except Exception:
            await index.index.delete()
            raise

        self.cache[vector_db.identifier] = index

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        if vector_db_id in self.cache:
            await self.cache[vector_db_id].index.delete()
            del self.cache[vector_db_id]

        assert self.kvstore is not None
        await self.kvstore.delete(key=f"{VECTOR_DBS_PREFIX}{vector_db_id}")

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        await index.insert_chunks(chunks)

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        index = await self._get_and_cache_vector_db_index(vector_db_id)
        return await index.query_chunks(query, params)

    async def _get_and_cache_vector_db_index(self, vector_db_id: str) -> VectorDBWithIndex:
        if vector_db_id in self.cache:
            return self.cache[vector_db_id]

        if self.vector_db_store is None:
            raise RuntimeError("Vector DB store not initialized")

        vector_db = self.vector_db_store.get_vector_db(vector_db_id)
        if vector_db is None:
            raise VectorStoreNotFoundError(vector_db_id)

        index = OpenGaussIndex(vector_db, vector_db.embedding_dimension, self.config)
        self.cache[vector_db_id] = VectorDBWithIndex(vector_db, index, self.inference_api)
        return self.cache[vector_db_id]

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        """Delete chunks from an OpenGauss vector store."""
        index = await self._get_and_cache_vector_db_index(store_id)
        if not index:
            raise VectorStoreNotFoundError(store_id)

        await index.index.delete_chunks(chunks_for_deletion)
