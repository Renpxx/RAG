"""
Central configuration helpers for the technical-document RAG pipeline.

All sensitive settings are read from environment variables to avoid
hard-coding provider credentials in source control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RAGSettings:
    """Container for all tunable pipeline settings."""

    api_type: str
    chat_base_url: str
    chat_api_key: str
    chat_model: str
    chat_temperature: float
    embedding_base_url: str
    embedding_api_key: str
    embedding_model: str
    embedding_batch_size: int
    prompt_template_path: str
    input_dir: str
    url_manifest: Optional[str]
    chunk_size: int
    chunk_overlap: int
    retriever_k: int
    persist_directory: str
    collection_name: str
    evaluation_samples: int
    evaluation_output_path: str
    langsmith_api_key: str
    langsmith_tracing: bool
    min_chunk_chars: int
    multi_query_variations: int
    enable_compression: bool
    enable_bm25: bool
    hybrid_search_ratio: float
    rerank_top_k: int


def _env(key: str, default: str = "") -> str:
    """Fetch an environment variable with a default fallback."""
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    """Fetch an integer environment variable."""
    try:
        return int(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    """Fetch a float environment variable."""
    try:
        return float(os.getenv(key, default))
    except (TypeError, ValueError):
        return default


def _candidate_url_manifest(input_dir: str) -> Optional[str]:
    """Return a default URL manifest path when available."""
    manifest = os.getenv("RAG_URL_MANIFEST")
    if manifest:
        return manifest
    candidate = Path(input_dir) / "urls.txt"
    return str(candidate) if candidate.exists() else None


def load_settings() -> RAGSettings:
    """
    Build the unified settings object from environment variables.

    Env vars follow the `RAG_*` prefix to keep them scoped to this project.
    """

    input_dir = _env("RAG_INPUT_DIR", "input")
    url_manifest = _candidate_url_manifest(input_dir)

    settings = RAGSettings(
        api_type=_env("RAG_API_TYPE", "openai"),
        chat_base_url=_env("RAG_CHAT_BASE_URL", "https://c-z0-api-01.hash070.com/v1"),
        chat_api_key=_env("RAG_CHAT_API_KEY", "*"),
        chat_model=_env("RAG_CHAT_MODEL", "gpt-5-nano"),
        chat_temperature=_env_float("RAG_CHAT_TEMPERATURE", 1.0),
        embedding_base_url=_env("RAG_EMBEDDING_BASE_URL", _env("RAG_CHAT_BASE_URL", "https://c-z0-api-01.hash070.com/v1")),
        embedding_api_key=_env("RAG_EMBEDDING_API_KEY", _env("RAG_CHAT_API_KEY", "*")),
        embedding_model=_env("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        embedding_batch_size=_env_int("RAG_EMBEDDING_BATCH_SIZE", 32),
        prompt_template_path=_env("RAG_PROMPT_TEMPLATE", "prompt_template.txt"),
        input_dir=input_dir,
        url_manifest=url_manifest,
        chunk_size=_env_int("RAG_CHUNK_SIZE", 1200),
        chunk_overlap=_env_int("RAG_CHUNK_OVERLAP", 200),
        retriever_k=_env_int("RAG_RETRIEVAL_K", 5),
        persist_directory=_env("RAG_VECTOR_DIR", "chromaDB"),
        collection_name=_env("RAG_COLLECTION", "technical_docs"),
        evaluation_samples=_env_int("RAG_EVAL_SAMPLES", 5),
        evaluation_output_path=_env(
            "RAG_EVAL_OUTPUT", "evaluation_results.json"
        ),
        langsmith_api_key=_env("LANGCHAIN_API_KEY", ""),
        langsmith_tracing=_env("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        min_chunk_chars=_env_int("RAG_MIN_CHUNK_CHARS", 200),
        multi_query_variations=_env_int("RAG_MULTI_QUERY_VARIATIONS", 2),
        enable_compression=_env("RAG_ENABLE_COMPRESSION", "true").lower() == "true",
        enable_bm25=_env("RAG_ENABLE_BM25", "true").lower() == "true",
        hybrid_search_ratio=_env_float("RAG_HYBRID_SEARCH_RATIO", 0.5),
        rerank_top_k=_env_int("RAG_RERANK_TOP_K", 10),
    )
    return settings
