# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging
import re
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
from llama_stack.providers.datatypes import Api, VectorDBsProtocolPrivate
from llama_stack.providers.utils.kvstore import kvstore_impl
from llama_stack.providers.utils.kvstore.api import KVStore
from llama_stack.providers.utils.memory.openai_vector_store_mixin import OpenAIVectorStoreMixin
from llama_stack.providers.utils.memory.vector_store import (
    RERANKER_TYPE_WEIGHTED,
    ChunkForDeletion,
    EmbeddingIndex,
    VectorDBWithIndex,
)

from .config import PGVectorVectorIOConfig

log = logging.getLogger(__name__)

VERSION = "v4"
VECTOR_DBS_PREFIX = f"vector_dbs:pgvector:{VERSION}::"
VECTOR_INDEX_PREFIX = f"vector_index:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_PREFIX = f"openai_vector_stores:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_PREFIX = f"openai_vector_stores_files:pgvector:{VERSION}::"
OPENAI_VECTOR_STORES_FILES_CONTENTS_PREFIX = f"openai_vector_stores_files_contents:pgvector:{VERSION}::"


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
        raise ValueError(f"Unsupported PostgreSQL FTS config: {name}")
    return f"'{name}'"


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", value or ""))


def check_extension_version(cur):
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    result = cur.fetchone()
    return result[0] if result else None


def upsert_models(conn, keys_models: list[tuple[str, BaseModel]]):
    with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
        query = sql.SQL(
            """
            INSERT INTO metadata_store (key, data)
            VALUES %s
            ON CONFLICT (key) DO UPDATE
            SET data = EXCLUDED.data
            """
        )

        values = [(key, Json(model.model_dump())) for key, model in keys_models]
        execute_values(cur, query, values, template="(%s, %s)")


def connect_pgvector(config: PGVectorVectorIOConfig):
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


class PGVectorIndex(EmbeddingIndex):
    def __init__(self, vector_db: VectorDB, dimension: int, conn, config: PGVectorVectorIOConfig, kvstore: KVStore | None = None):
        self.conn = conn
        self.config = config
        sanitized_identifier = _make_sql_identifier(vector_db.identifier)
        self.table_name = f"vector_store_{sanitized_identifier}"
        self.fts_index_name = _make_index_name(self.table_name, "content_fts_gin_idx")
        self.fts_config_literal = _make_regconfig_literal(self.config.fts_config)
        self.kvstore = kvstore

        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    document JSONB,
                    embedding vector({dimension})
                )
                """
            )
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.fts_index_name}
                ON {self.table_name}
                USING gin (to_tsvector({self.fts_config_literal}, coalesce(content, '')))
                """
            )

    async def add_chunks(self, chunks: list[Chunk], embeddings: NDArray):
        assert len(chunks) == len(embeddings), (
            f"Chunk length {len(chunks)} does not match embedding length {len(embeddings)}"
        )
        assert all(isinstance(chunk.content, str) for chunk in chunks), "PGVector keyword search only supports text chunks"

        values = []
        for i, chunk in enumerate(chunks):
            values.append(
                (
                    f"{chunk.chunk_id}",
                    chunk.content,
                    Json(chunk.model_dump()),
                    embeddings[i].tolist(),
                )
            )

        query = sql.SQL(
            f"""
            INSERT INTO {self.table_name} (id, content, document, embedding)
            VALUES %s
            ON CONFLICT (id) DO UPDATE
            SET content = EXCLUDED.content,
                document = EXCLUDED.document,
                embedding = EXCLUDED.embedding
            """
        )
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            execute_values(cur, query, values, template="(%s, %s, %s, %s::vector)")

    async def query_vector(self, embedding: NDArray, k: int, score_threshold: float) -> QueryChunksResponse:
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
                SELECT document, embedding <=> %s::vector AS distance
                FROM {self.table_name}
                ORDER BY distance
                LIMIT %s
                """,
                (embedding.tolist(), k),
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

        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(
                f"""
                SELECT document,
                       ts_rank(
                           to_tsvector({self.fts_config_literal}, coalesce(content, '')),
                           plainto_tsquery({self.fts_config_literal}, %s)
                       ) AS score
                FROM {self.table_name}
                WHERE to_tsvector({self.fts_config_literal}, coalesce(content, ''))
                      @@ plainto_tsquery({self.fts_config_literal}, %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query_string, query_string, k),
            )
            results = cur.fetchall()

            if not results and _contains_cjk(query_string):
                cur.execute(
                    f"""
                    SELECT document,
                           (
                               CASE
                                   WHEN position(lower(%s) in lower(coalesce(content, ''))) > 0
                                   THEN 1.0 / position(lower(%s) in lower(coalesce(content, '')))
                                   ELSE 0.0
                               END
                           ) +
                           (
                               (
                                   length(lower(coalesce(content, '')))
                                   - length(replace(lower(coalesce(content, '')), lower(%s), ''))
                               ) / GREATEST(length(%s), 1)
                           ) AS score
                    FROM {self.table_name}
                    WHERE lower(coalesce(content, '')) LIKE '%%' || lower(%s) || '%%'
                    ORDER BY score DESC
                    LIMIT %s
                    """,
                    (query_string, query_string, query_string, query_string, query_string, k),
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
            chunk.chunk_id: score for chunk, score in zip(keyword_response.chunks, keyword_response.scores, strict=False)
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
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    async def delete_chunks(self, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        chunk_ids = [c.chunk_id for c in chunks_for_deletion]
        with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (chunk_ids,))


class PGVectorVectorIOAdapter(OpenAIVectorStoreMixin, VectorIO, VectorDBsProtocolPrivate):
    def __init__(
        self,
        config: PGVectorVectorIOConfig,
        inference_api: Api.inference,
        files_api: Files | None = None,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.conn = None
        self.cache: dict[str, VectorDBWithIndex] = {}
        self.files_api = files_api
        self.kvstore: KVStore | None = None
        self.vector_db_store = None
        self.openai_vector_store: dict[str, dict[str, Any]] = {}
        self.metadatadata_collection_name = "openai_vector_stores_metadata"

    async def initialize(self) -> None:
        log.info(f"Initializing PGVector memory adapter with config: {self.config}")
        self.kvstore = await kvstore_impl(self.config.kvstore)
        start_key = VECTOR_DBS_PREFIX
        end_key = f"{VECTOR_DBS_PREFIX}\xff"
        stored_vector_dbs = [VectorDB.model_validate_json(raw) for raw in await self.kvstore.values_in_range(start_key, end_key)]
        await self.initialize_openai_vector_stores()

        try:
            self.conn = connect_pgvector(self.config)
            with self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                version = check_extension_version(cur)
                if version:
                    log.info(f"Vector extension version: {version}")
                else:
                    raise RuntimeError("Vector extension is not installed.")

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metadata_store (
                        key TEXT PRIMARY KEY,
                        data JSONB
                    )
                    """
                )

            for vector_db in stored_vector_dbs:
                self.cache[vector_db.identifier] = VectorDBWithIndex(
                    vector_db=vector_db,
                    index=PGVectorIndex(
                        vector_db,
                        vector_db.embedding_dimension,
                        self.conn,
                        self.config,
                        kvstore=self.kvstore,
                    ),
                    inference_api=self.inference_api,
                )
        except Exception as e:
            log.exception("Could not connect to PGVector database server")
            raise RuntimeError("Could not connect to PGVector database server") from e

    async def shutdown(self) -> None:
        if self.conn is not None:
            self.conn.close()
            log.info("Connection to PGVector database server closed")

    async def register_vector_db(self, vector_db: VectorDB) -> None:
        assert self.kvstore is not None
        upsert_models(self.conn, [(vector_db.identifier, vector_db)])
        await self.kvstore.set(key=f"{VECTOR_DBS_PREFIX}{vector_db.identifier}", value=vector_db.model_dump_json())

        index = VectorDBWithIndex(
            vector_db,
            index=PGVectorIndex(
                vector_db,
                vector_db.embedding_dimension,
                self.conn,
                self.config,
                kvstore=self.kvstore,
            ),
            inference_api=self.inference_api,
        )
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

        vector_db = await self.vector_db_store.get_vector_db(vector_db_id)
        if vector_db is None:
            raise VectorStoreNotFoundError(vector_db_id)

        index = PGVectorIndex(vector_db, vector_db.embedding_dimension, self.conn, self.config)
        self.cache[vector_db_id] = VectorDBWithIndex(vector_db, index, self.inference_api)
        return self.cache[vector_db_id]

    async def delete_chunks(self, store_id: str, chunks_for_deletion: list[ChunkForDeletion]) -> None:
        index = await self._get_and_cache_vector_db_index(store_id)
        if not index:
            raise VectorStoreNotFoundError(store_id)

        await index.index.delete_chunks(chunks_for_deletion)
