# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Any

from pydantic import BaseModel, Field

from llama_stack.providers.utils.kvstore.config import (
    KVStoreConfig,
    SqliteKVStoreConfig,
)
from llama_stack.schema_utils import json_schema_type


@json_schema_type
class OpenGaussVectorIOConfig(BaseModel):
    host: str | None = Field(default="localhost")
    port: int | None = Field(default=5432)
    db: str | None = Field(default="postgres")
    user: str | None = Field(default="postgres")
    password: str | None = Field(default="mysecretpassword")
    connect_timeout: int = Field(default=5, ge=1, le=300)
    hnsw_m: int = Field(default=16, ge=2, le=100)
    hnsw_ef_construction: int = Field(default=200, ge=4, le=1000)
    hnsw_ef_search: int = Field(default=40, ge=1, le=1000)
    fts_config: str = Field(default="pg_catalog.english")
    kvstore: KVStoreConfig | None = Field(description="Config for KV store backend (SQLite only for now)", default=None)

    @classmethod
    def sample_run_config(
        cls,
        __distro_dir__: str,
        host: str = "${env.OPENGAUSS_HOST:=localhost}",
        port: str = "${env.OPENGAUSS_PORT:=5432}",
        db: str = "${env.OPENGAUSS_DB}",
        user: str = "${env.OPENGAUSS_USER}",
        password: str = "${env.OPENGAUSS_PASSWORD}",
        connect_timeout: str = "${env.OPENGAUSS_CONNECT_TIMEOUT:=5}",
        hnsw_m: str = "${env.OPENGAUSS_HNSW_M:=16}",
        hnsw_ef_construction: str = "${env.OPENGAUSS_HNSW_EF_CONSTRUCTION:=200}",
        hnsw_ef_search: str = "${env.OPENGAUSS_HNSW_EF_SEARCH:=40}",
        fts_config: str = "${env.OPENGAUSS_FTS_CONFIG:=pg_catalog.english}",
        **kwargs: Any,
    ) -> dict[str, Any]:
        return {
            "host": host,
            "port": port,
            "db": db,
            "user": user,
            "password": password,
            "connect_timeout": connect_timeout,
            "hnsw_m": hnsw_m,
            "hnsw_ef_construction": hnsw_ef_construction,
            "hnsw_ef_search": hnsw_ef_search,
            "fts_config": fts_config,
            "kvstore": SqliteKVStoreConfig.sample_run_config(
                __distro_dir__=__distro_dir__,
                db_name="opengauss_registry.db",
            ),
        }
