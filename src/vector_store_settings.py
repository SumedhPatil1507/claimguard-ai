"""
vector_store_settings.py
========================
Pydantic v2 BaseSettings for the PolicyVectorStore.

All values have sensible defaults so the store works out-of-the-box without
any environment variables.  Override any field by setting the corresponding
env-var (prefix: CLAIMGUARD_VS_) or by instantiating VectorStoreSettings
directly in tests.

Environment variable examples
------------------------------
    CLAIMGUARD_VS_QDRANT_MODE=remote
    CLAIMGUARD_VS_QDRANT_URL=http://localhost:6333
    CLAIMGUARD_VS_QDRANT_API_KEY=my-secret-key
    CLAIMGUARD_VS_RERANKER_BACKEND=cross_encoder
    CLAIMGUARD_VS_COHERE_API_KEY=co-xxxxx
    CLAIMGUARD_VS_TOP_K_DENSE=10
    CLAIMGUARD_VS_TOP_K_BM25=10
    CLAIMGUARD_VS_TOP_K_RERANK=3
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums for constrained string fields
# ---------------------------------------------------------------------------


class QdrantMode(str, Enum):
    """Where to store vectors."""
    memory = "memory"       # :memory: — no persistence, useful for tests
    local = "local"         # local file-based storage (Qdrant embedded)
    remote = "remote"       # connect to a running Qdrant server


class RerankerBackend(str, Enum):
    """Which re-ranker to use after hybrid retrieval."""
    none = "none"               # skip re-ranking; return fused scores as-is
    cross_encoder = "cross_encoder"   # sentence-transformers CrossEncoder (local)
    cohere = "cohere"           # Cohere Rerank API (requires COHERE_API_KEY)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class VectorStoreSettings(BaseSettings):
    """
    All configuration for PolicyVectorStore.

    Loaded from environment variables with the prefix ``CLAIMGUARD_VS_``.
    Defaults allow the store to run completely offline without any config.
    """

    model_config = SettingsConfigDict(
        env_prefix="CLAIMGUARD_VS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant connection ────────────────────────────────────────────────────

    qdrant_mode: QdrantMode = Field(
        default=QdrantMode.local,
        description=(
            "Storage mode: 'memory' (ephemeral, for tests), "
            "'local' (on-disk embedded), or 'remote' (Qdrant server)."
        ),
    )

    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL.  Only used when qdrant_mode='remote'.",
    )

    qdrant_api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for authenticated cloud clusters.",
    )

    qdrant_collection: str = Field(
        default="policy_docs",
        description="Name of the Qdrant collection to create / query.",
    )

    qdrant_local_path: Path = Field(
        default=Path("data/qdrant_db"),
        description="Directory for on-disk Qdrant storage (qdrant_mode='local').",
    )

    # ── Embedding model ──────────────────────────────────────────────────────

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model name for dense embeddings.",
    )

    embedding_dim: int = Field(
        default=384,
        ge=1,
        description=(
            "Dimension of the dense embedding vectors.  "
            "Must match the chosen embedding_model output dimension."
        ),
    )

    # ── Chunking ─────────────────────────────────────────────────────────────

    chunk_size: int = Field(
        default=300,
        ge=50,
        description="Character count per chunk (before overlap).",
    )

    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Overlap in characters between consecutive chunks.",
    )

    # ── Retrieval counts ─────────────────────────────────────────────────────

    top_k_dense: int = Field(
        default=10,
        ge=1,
        description="Number of candidates to fetch from the Qdrant dense search.",
    )

    top_k_bm25: int = Field(
        default=10,
        ge=1,
        description="Number of candidates to fetch from the BM25 sparse search.",
    )

    top_k_rerank: int = Field(
        default=3,
        ge=1,
        description="Number of results to return after re-ranking / fusion.",
    )

    # ── Re-ranking ───────────────────────────────────────────────────────────

    reranker_backend: RerankerBackend = Field(
        default=RerankerBackend.cross_encoder,
        description=(
            "Re-ranker backend: 'none' (RRF fusion only), "
            "'cross_encoder' (local sentence-transformers), "
            "or 'cohere' (Cohere Rerank API)."
        ),
    )

    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="sentence-transformers CrossEncoder model name.",
    )

    cohere_api_key: Optional[str] = Field(
        default=None,
        description="Cohere API key.  Required when reranker_backend='cohere'.",
    )

    cohere_rerank_model: str = Field(
        default="rerank-english-v3.0",
        description="Cohere rerank model identifier.",
    )

    # ── BM25 ─────────────────────────────────────────────────────────────────

    bm25_k1: float = Field(
        default=1.5,
        ge=0.0,
        description="BM25 k1 term-frequency saturation parameter.",
    )

    bm25_b: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="BM25 b document-length normalisation parameter.",
    )

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────

    rrf_k: int = Field(
        default=60,
        ge=1,
        description="RRF constant k.  Typical value: 60.",
    )

    # ── Docs directory ───────────────────────────────────────────────────────

    docs_dir: Path = Field(
        default=Path("data/policy_docs"),
        description="Directory containing .txt policy / IRDAI documents to index.",
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:  # type: ignore[override]
        chunk_size = info.data.get("chunk_size", 300)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})."
            )
        return v

    @field_validator("top_k_rerank")
    @classmethod
    def rerank_le_candidates(cls, v: int, info) -> int:  # type: ignore[override]
        max_candidates = max(
            info.data.get("top_k_dense", 10),
            info.data.get("top_k_bm25", 10),
        )
        if v > max_candidates:
            raise ValueError(
                f"top_k_rerank ({v}) cannot exceed max(top_k_dense, top_k_bm25) ({max_candidates})."
            )
        return v
