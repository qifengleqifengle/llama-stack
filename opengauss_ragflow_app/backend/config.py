import os
from pathlib import Path

from pydantic import BaseModel, Field


APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
FRONTEND_DIR = APP_ROOT / "frontend"


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    value = _env_str(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = _env_str(name)
    return float(value) if value is not None else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env_str(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


class AppSettings(BaseModel):
    llama_stack_base_url: str = Field(default="http://localhost:8321")
    openai_base_url: str = Field(default="http://localhost:8000/v1")
    openai_api_key: str | None = None
    openai_model: str | None = None
    openai_request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=512, ge=1, le=8192)

    mineru_base_url: str = Field(default="https://mineru.net")
    mineru_api_token: str | None = None
    mineru_user_token: str | None = None
    mineru_model_version: str = Field(default="vlm")
    mineru_html_model_version: str = Field(default="MinerU-HTML")
    mineru_language: str = Field(default="auto")
    mineru_enable_formula: bool = Field(default=True)
    mineru_enable_table: bool = Field(default=True)
    mineru_enable_ocr: bool = Field(default=False)
    mineru_request_timeout_seconds: float = Field(default=60.0, gt=0.0)
    mineru_poll_interval_seconds: float = Field(default=2.0, gt=0.0)
    mineru_poll_timeout_seconds: float = Field(default=300.0, gt=0.0)

    default_provider_id: str = Field(default="remote::pgvector")
    default_chunk_size: int = Field(default=512, ge=64, le=4096)
    default_retrieval_k: int = Field(default=6, ge=1, le=20)
    default_chat_k: int = Field(default=5, ge=1, le=20)
    default_mode: str = Field(default="hybrid")
    default_query_rewrite: bool = Field(default=False)
    default_ranker: str = Field(default="rrf")
    default_ranker_alpha: float = Field(default=0.6, ge=0.0, le=1.0)
    default_ranker_impact_factor: float = Field(default=60.0, gt=0.0)
    pgvector_host: str = Field(default="localhost")
    pgvector_port: int = Field(default=5432, gt=0)
    pgvector_db: str = Field(default="postgres")
    pgvector_user: str = Field(default="postgres")
    pgvector_password: str | None = None
    pgvector_connect_timeout: int = Field(default=5, gt=0)
    catalog_path: Path = Field(default=DATA_DIR / "catalog.json")


def load_settings() -> AppSettings:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return AppSettings(
        llama_stack_base_url=_env_str("OPENGAUSS_RAGFLOW_LLAMA_STACK_BASE_URL", "http://localhost:8321") or "http://localhost:8321",
        openai_base_url=_env_str("OPENGAUSS_RAGFLOW_OPENAI_BASE_URL", "http://localhost:8000/v1") or "http://localhost:8000/v1",
        openai_api_key=_env_str("OPENGAUSS_RAGFLOW_OPENAI_API_KEY"),
        openai_model=_env_str("OPENGAUSS_RAGFLOW_OPENAI_MODEL"),
        openai_request_timeout_seconds=_env_float("OPENGAUSS_RAGFLOW_OPENAI_TIMEOUT_SECONDS", 180.0),
        openai_temperature=_env_float("OPENGAUSS_RAGFLOW_OPENAI_TEMPERATURE", 0.2),
        openai_max_tokens=_env_int("OPENGAUSS_RAGFLOW_OPENAI_MAX_TOKENS", 512),
        mineru_base_url=_env_str("OPENGAUSS_RAGFLOW_MINERU_BASE_URL", "https://mineru.net") or "https://mineru.net",
        mineru_api_token=_env_str("OPENGAUSS_RAGFLOW_MINERU_API_TOKEN"),
        mineru_user_token=_env_str("OPENGAUSS_RAGFLOW_MINERU_USER_TOKEN"),
        mineru_model_version=_env_str("OPENGAUSS_RAGFLOW_MINERU_MODEL_VERSION", "vlm") or "vlm",
        mineru_html_model_version=_env_str("OPENGAUSS_RAGFLOW_MINERU_HTML_MODEL_VERSION", "MinerU-HTML") or "MinerU-HTML",
        mineru_language=_env_str("OPENGAUSS_RAGFLOW_MINERU_LANGUAGE", "auto") or "auto",
        mineru_enable_formula=_env_bool("OPENGAUSS_RAGFLOW_MINERU_ENABLE_FORMULA", True),
        mineru_enable_table=_env_bool("OPENGAUSS_RAGFLOW_MINERU_ENABLE_TABLE", True),
        mineru_enable_ocr=_env_bool("OPENGAUSS_RAGFLOW_MINERU_ENABLE_OCR", False),
        mineru_request_timeout_seconds=_env_float("OPENGAUSS_RAGFLOW_MINERU_TIMEOUT_SECONDS", 60.0),
        mineru_poll_interval_seconds=_env_float("OPENGAUSS_RAGFLOW_MINERU_POLL_INTERVAL_SECONDS", 2.0),
        mineru_poll_timeout_seconds=_env_float("OPENGAUSS_RAGFLOW_MINERU_POLL_TIMEOUT_SECONDS", 300.0),
        default_provider_id=_env_str("OPENGAUSS_RAGFLOW_DEFAULT_PROVIDER_ID", "remote::pgvector") or "remote::pgvector",
        default_chunk_size=_env_int("OPENGAUSS_RAGFLOW_DEFAULT_CHUNK_SIZE", 512),
        default_retrieval_k=_env_int("OPENGAUSS_RAGFLOW_DEFAULT_RETRIEVAL_K", 6),
        default_chat_k=_env_int("OPENGAUSS_RAGFLOW_DEFAULT_CHAT_K", 5),
        default_mode=_env_str("OPENGAUSS_RAGFLOW_DEFAULT_MODE", "hybrid") or "hybrid",
        default_query_rewrite=_env_bool("OPENGAUSS_RAGFLOW_DEFAULT_QUERY_REWRITE", False),
        default_ranker=_env_str("OPENGAUSS_RAGFLOW_DEFAULT_RANKER", "rrf") or "rrf",
        default_ranker_alpha=_env_float("OPENGAUSS_RAGFLOW_DEFAULT_RANKER_ALPHA", 0.6),
        default_ranker_impact_factor=_env_float("OPENGAUSS_RAGFLOW_DEFAULT_RANKER_IMPACT_FACTOR", 60.0),
        pgvector_host=_env_str("OPENGAUSS_RAGFLOW_PGVECTOR_HOST", _env_str("PGVECTOR_HOST", "localhost")) or "localhost",
        pgvector_port=_env_int("OPENGAUSS_RAGFLOW_PGVECTOR_PORT", _env_int("PGVECTOR_PORT", 5432)),
        pgvector_db=_env_str("OPENGAUSS_RAGFLOW_PGVECTOR_DB", _env_str("PGVECTOR_DB", "postgres")) or "postgres",
        pgvector_user=_env_str("OPENGAUSS_RAGFLOW_PGVECTOR_USER", _env_str("PGVECTOR_USER", "postgres")) or "postgres",
        pgvector_password=_env_str("OPENGAUSS_RAGFLOW_PGVECTOR_PASSWORD", _env_str("PGVECTOR_PASSWORD")),
        pgvector_connect_timeout=_env_int(
            "OPENGAUSS_RAGFLOW_PGVECTOR_CONNECT_TIMEOUT",
            _env_int("PGVECTOR_CONNECT_TIMEOUT", 5),
        ),
        catalog_path=Path(_env_str("OPENGAUSS_RAGFLOW_CATALOG_PATH", str(DATA_DIR / "catalog.json")) or str(DATA_DIR / "catalog.json")),
    )
