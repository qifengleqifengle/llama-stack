# remote::opengauss

## Description


[OpenGauss](https://opengauss.org/en/) is a remote vector database provider for Llama Stack. It
allows you to store and query vectors directly in memory.
That means you'll get fast and efficient vector retrieval.

## Features

- Easy to use
- Fully integrated with Llama Stack
- Vector search with OpenGauss DataVec and HNSW indexes on a row-store table
- Keyword search with OpenGauss FTS and GIN indexes on the content column
- Hybrid search with RRF or weighted reranking across vector and keyword retrieval

## Usage

To use OpenGauss in your Llama Stack project, follow these steps:

1. Install the necessary dependencies.
2. Configure your Llama Stack project to use OpenGauss.
3. Start storing and querying vectors.

The provider uses a single row-store table per registered vector DB:

- `content` is indexed with `GIN(to_tsvector(...))` for keyword retrieval
- `embedding` is indexed with `HNSW` when the embedding dimension is within the OpenGauss index limit

When the embedding dimension exceeds the OpenGauss HNSW limit, vector queries automatically fall back to exact distance search.

## Installation

You can install OpenGauss using docker:

```bash
docker pull opengauss/opengauss:latest
```
## Documentation
See [OpenGauss' documentation](https://docs.opengauss.org/en/docs/5.0.0/docs/GettingStarted/understanding-opengauss.html) for more details about OpenGauss in general.


## Configuration

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `host` | `str \| None` | No | localhost |  |
| `port` | `int \| None` | No | 5432 |  |
| `db` | `str \| None` | No | postgres |  |
| `user` | `str \| None` | No | postgres |  |
| `password` | `str \| None` | No | mysecretpassword |  |
| `hnsw_m` | `int` | No | 16 | HNSW graph degree used when building the vector index |
| `hnsw_ef_construction` | `int` | No | 200 | HNSW construction search width |
| `hnsw_ef_search` | `int` | No | 40 | HNSW search breadth used during vector queries |
| `fts_config` | `str` | No | pg_catalog.english | OpenGauss text search configuration used by `to_tsvector` and `plainto_tsquery` |
| `kvstore` | `utils.kvstore.config.RedisKVStoreConfig \| utils.kvstore.config.SqliteKVStoreConfig \| utils.kvstore.config.PostgresKVStoreConfig \| utils.kvstore.config.MongoDBKVStoreConfig, annotation=NoneType, required=False, default='sqlite', discriminator='type'` | No |  | Config for KV store backend (SQLite only for now) |

## Sample Configuration

```yaml
host: ${env.OPENGAUSS_HOST:=localhost}
port: ${env.OPENGAUSS_PORT:=5432}
db: ${env.OPENGAUSS_DB}
user: ${env.OPENGAUSS_USER}
password: ${env.OPENGAUSS_PASSWORD}
hnsw_m: ${env.OPENGAUSS_HNSW_M:=16}
hnsw_ef_construction: ${env.OPENGAUSS_HNSW_EF_CONSTRUCTION:=200}
hnsw_ef_search: ${env.OPENGAUSS_HNSW_EF_SEARCH:=40}
fts_config: ${env.OPENGAUSS_FTS_CONFIG:=pg_catalog.english}
kvstore:
  type: sqlite
  db_path: ${env.SQLITE_STORE_DIR:=~/.llama/dummy}/opengauss_registry.db

```
