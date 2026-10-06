"""
vector_store.py
===============
Hybrid (dense + BM25 sparse) vector store for the ClaimGuard AI Policy Copilot.

Architecture
------------
1. **Dense retrieval** — sentence-transformers all-MiniLM-L6-v2 embeddings
   stored and queried via Qdrant (in-memory, local on-disk, or remote server).
2. **Sparse retrieval** — BM25 index (rank-bm25) built over the same corpus
   and queried in-process.
3. **Fusion** — Reciprocal Rank Fusion (RRF) merges both ranked lists into a
   single candidate set.
4. **Re-ranking** — optional second-pass scoring using either:
   * a local sentence-transformers CrossEncoder  (default, no API key needed)
   * the Cohere Rerank API                       (requires COHERE_API_KEY)
   * none                                        (return RRF scores as-is)

Graceful degradation
--------------------
Every optional dependency is guarded by a try/except at import time.
The store is always safe to import.  Missing dependencies reduce quality
but never raise exceptions:

  | Missing package           | Effect                               |
  |---------------------------|--------------------------------------|
  | qdrant-client             | no dense retrieval; BM25 only        |
  | sentence-transformers     | no dense embeddings or cross-encoder |
  | rank-bm25                 | no sparse retrieval; dense only      |
  | cohere                    | falls back to cross-encoder or none  |
  | all retrieval unavailable | search() returns []                  |

Configuration
-------------
All tuneable parameters live in VectorStoreSettings (Pydantic BaseSettings).
Override via env-vars prefixed CLAIMGUARD_VS_ or pass a settings instance
directly to PolicyVectorStore().

Usage
-----
    from src.vector_store import policy_store

    hits = policy_store.search("IRDAI claim settlement 30 days", top_k=3)
    # [{"content": "...", "source": "irdai_guidelines.txt", "score": 0.91}, ...]
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.vector_store_settings import (
    QdrantMode,
    RerankerBackend,
    VectorStoreSettings,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

# ── Qdrant ──────────────────────────────────────────────────────────────────
try:
    from qdrant_client import QdrantClient                          # type: ignore
    from qdrant_client.models import (                              # type: ignore
        Distance,
        PointStruct,
        VectorParams,
    )
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

# ── Sentence-transformers (dense embedder + cross-encoder reranker) ──────────
try:
    from sentence_transformers import SentenceTransformer           # type: ignore
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from sentence_transformers.cross_encoder import CrossEncoder    # type: ignore
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

# ── BM25 ────────────────────────────────────────────────────────────────────
try:
    from rank_bm25 import BM25Okapi                                 # type: ignore
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# ── Cohere ──────────────────────────────────────────────────────────────────
try:
    import cohere                                                    # type: ignore
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

# A single document chunk as stored internally.
_Chunk = Dict[str, str]          # {"content": str, "source": str}

# A ranked hit returned to callers.
SearchResult = Dict[str, Any]    # {"content", "source", "score"}


# ---------------------------------------------------------------------------
# PolicyVectorStore
# ---------------------------------------------------------------------------


class PolicyVectorStore:
    """
    Hybrid dense + BM25 retriever with optional cross-encoder re-ranking.

    All heavy work (model loading, indexing) is deferred to the first
    call to search() or an explicit build_index() call.

    Parameters
    ----------
    settings : VectorStoreSettings, optional
        Override the default configuration.  Useful in tests.
    """

    def __init__(self, settings: Optional[VectorStoreSettings] = None) -> None:
        self._cfg = settings or VectorStoreSettings()
        self._built = False

        # Dense retrieval state
        self._qdrant: Optional[Any] = None          # QdrantClient
        self._embedder: Optional[Any] = None        # SentenceTransformer

        # Sparse retrieval state
        self._bm25_index: Optional[Any] = None      # BM25Okapi
        self._corpus: List[_Chunk] = []             # all chunks (shared)
        self._tokenised_corpus: List[List[str]] = []

        # Re-ranking state
        self._cross_encoder: Optional[Any] = None   # CrossEncoder
        self._cohere_client: Optional[Any] = None   # cohere.Client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Load documents, build dense Qdrant index and BM25 index.
        Safe to call multiple times (no-op after first successful build).
        """
        if self._built:
            return
        self._built = True  # set early to prevent re-entrant calls

        chunks = self._load_and_chunk_all()
        if not chunks:
            logger.warning(
                "vector_store: no documents found in '%s'; retrieval will return [].",
                self._cfg.docs_dir,
            )
            return

        self._corpus = chunks
        self._tokenised_corpus = [self._tokenise(c["content"]) for c in chunks]

        self._init_bm25()
        self._init_dense(chunks)
        self._init_reranker()

        logger.info(
            "vector_store: indexed %d chunks | dense=%s bm25=%s reranker=%s",
            len(chunks),
            "✓" if self._qdrant else "✗",
            "✓" if self._bm25_index else "✗",
            self._cfg.reranker_backend.value,
        )

    def search(self, query: str, top_k: Optional[int] = None) -> List[SearchResult]:
        """
        Hybrid search: dense + BM25 → RRF fusion → optional re-ranking.

        Parameters
        ----------
        query : str
            Natural-language query.
        top_k : int, optional
            Number of results to return.  Defaults to settings.top_k_rerank.

        Returns
        -------
        list of dicts with keys: content, source, score
        Empty list if nothing is indexed or an error occurs.
        """
        if not self._built:
            self.build_index()

        if not self._corpus:
            return []

        k = top_k if top_k is not None else self._cfg.top_k_rerank

        try:
            dense_hits = self._dense_search(query, self._cfg.top_k_dense)
            bm25_hits  = self._bm25_search(query, self._cfg.top_k_bm25)
            fused      = self._rrf_fuse(dense_hits, bm25_hits)
            reranked   = self._rerank(query, fused, k)
            return reranked[:k]
        except Exception as exc:
            logger.error("vector_store: search failed — %s", exc, exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    def _load_and_chunk_all(self) -> List[_Chunk]:
        """Read every .txt file in docs_dir and return all chunks."""
        docs_dir = Path(self._cfg.docs_dir)
        chunks: List[_Chunk] = []
        if not docs_dir.exists():
            logger.warning("vector_store: docs_dir '%s' does not exist.", docs_dir)
            return chunks
        for txt_file in sorted(docs_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding="utf-8")
                chunks.extend(self._chunk_text(text, source=txt_file.name))
            except Exception as exc:
                logger.warning("vector_store: could not read '%s' — %s", txt_file, exc)
        return chunks

    def _chunk_text(self, text: str, source: str) -> List[_Chunk]:
        """
        Sliding-window character-level chunker.

        Produces chunks of cfg.chunk_size chars with cfg.chunk_overlap
        chars of overlap between consecutive windows.
        """
        size    = self._cfg.chunk_size
        overlap = self._cfg.chunk_overlap
        step    = size - overlap
        chunks: List[_Chunk] = []
        start = 0
        while start < len(text):
            chunk = text[start : start + size].strip()
            if chunk:
                chunks.append({"content": chunk, "source": source})
            start += step
        return chunks

    # ------------------------------------------------------------------
    # BM25 initialisation & search
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> List[str]:
        """Lowercase whitespace tokeniser for BM25."""
        return text.lower().split()

    def _init_bm25(self) -> None:
        if not HAS_BM25:
            logger.warning("vector_store: rank-bm25 not installed; sparse retrieval disabled.")
            return
        try:
            self._bm25_index = BM25Okapi(
                self._tokenised_corpus,
                k1=self._cfg.bm25_k1,
                b=self._cfg.bm25_b,
            )
        except Exception as exc:
            logger.error("vector_store: BM25 init failed — %s", exc)

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (corpus_index, bm25_score) sorted descending.
        Empty list if BM25 not available.
        """
        if self._bm25_index is None:
            return []
        try:
            tokens = self._tokenise(query)
            scores: List[float] = self._bm25_index.get_scores(tokens).tolist()
            indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
            return indexed[:top_k]
        except Exception as exc:
            logger.error("vector_store: BM25 search failed — %s", exc)
            return []

    # ------------------------------------------------------------------
    # Dense (Qdrant) initialisation & search
    # ------------------------------------------------------------------

    def _init_dense(self, chunks: List[_Chunk]) -> None:
        if not HAS_QDRANT or not HAS_ST:
            logger.warning(
                "vector_store: qdrant-client or sentence-transformers not installed; "
                "dense retrieval disabled."
            )
            return
        try:
            self._embedder = SentenceTransformer(self._cfg.embedding_model)
            self._qdrant   = self._make_qdrant_client()
            self._ensure_collection()
            self._upsert_chunks(chunks)
        except Exception as exc:
            logger.error("vector_store: dense index init failed — %s", exc)
            self._qdrant   = None
            self._embedder = None

    def _make_qdrant_client(self) -> Any:
        """Construct a QdrantClient for the configured mode."""
        mode = self._cfg.qdrant_mode
        if mode == QdrantMode.memory:
            return QdrantClient(":memory:")
        if mode == QdrantMode.local:
            local_path = Path(self._cfg.qdrant_local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            return QdrantClient(path=str(local_path))
        # remote
        return QdrantClient(
            url=self._cfg.qdrant_url,
            api_key=self._cfg.qdrant_api_key,
        )

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it does not already exist."""
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self._cfg.qdrant_collection not in existing:
            self._qdrant.create_collection(
                collection_name=self._cfg.qdrant_collection,
                vectors_config=VectorParams(
                    size=self._cfg.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

    def _upsert_chunks(self, chunks: List[_Chunk]) -> None:
        """Embed all chunks and upsert into Qdrant."""
        texts      = [c["content"] for c in chunks]
        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()
        points = [
            PointStruct(
                id=idx,
                vector=emb,
                payload={"content": chunks[idx]["content"], "source": chunks[idx]["source"]},
            )
            for idx, emb in enumerate(embeddings)
        ]
        # Batch upsert in chunks of 256 to avoid large payloads.
        batch_size = 256
        for i in range(0, len(points), batch_size):
            self._qdrant.upsert(
                collection_name=self._cfg.qdrant_collection,
                points=points[i : i + batch_size],
            )

    def _dense_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """
        Returns list of (corpus_index, cosine_similarity) sorted descending.
        Empty list if Qdrant/embedder not available.
        """
        if self._qdrant is None or self._embedder is None:
            return []
        try:
            q_emb = self._embedder.encode([query], show_progress_bar=False).tolist()[0]
            hits  = self._qdrant.search(
                collection_name=self._cfg.qdrant_collection,
                query_vector=q_emb,
                limit=top_k,
                with_payload=False,
            )
            # Each hit.id corresponds to the chunk index in self._corpus.
            return [(int(h.id), float(h.score)) for h in hits]
        except Exception as exc:
            logger.error("vector_store: dense search failed — %s", exc)
            return []

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_hits: List[Tuple[int, float]],
        bm25_hits:  List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """
        Merge two ranked lists using Reciprocal Rank Fusion.

        RRF score = Σ  1 / (k + rank_i)   where k = cfg.rrf_k
        Returns (corpus_index, rrf_score) sorted descending.
        """
        k = self._cfg.rrf_k
        scores: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

        for rank, (idx, _) in enumerate(bm25_hits, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)

        if not scores:
            return []

        # Normalise to [0, 1] relative to the max observed RRF score.
        max_score = max(scores.values())
        normalised = {
            idx: score / max_score
            for idx, score in scores.items()
        }
        return sorted(normalised.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Re-ranking
    # ------------------------------------------------------------------

    def _init_reranker(self) -> None:
        backend = self._cfg.reranker_backend
        if backend == RerankerBackend.none:
            return

        if backend == RerankerBackend.cross_encoder:
            if not HAS_CROSS_ENCODER:
                logger.warning(
                    "vector_store: sentence-transformers CrossEncoder not available; "
                    "falling back to RRF-only ranking."
                )
                return
            try:
                self._cross_encoder = CrossEncoder(self._cfg.cross_encoder_model)
                logger.info(
                    "vector_store: CrossEncoder loaded (%s).",
                    self._cfg.cross_encoder_model,
                )
            except Exception as exc:
                logger.error("vector_store: CrossEncoder init failed — %s", exc)

        elif backend == RerankerBackend.cohere:
            if not HAS_COHERE:
                logger.warning(
                    "vector_store: cohere package not installed; "
                    "falling back to cross-encoder or RRF."
                )
                return
            api_key = self._cfg.cohere_api_key
            if not api_key:
                logger.warning(
                    "vector_store: CLAIMGUARD_VS_COHERE_API_KEY not set; "
                    "falling back to cross-encoder or RRF."
                )
                return
            try:
                self._cohere_client = cohere.Client(api_key)
                logger.info("vector_store: Cohere reranker initialised.")
            except Exception as exc:
                logger.error("vector_store: Cohere init failed — %s", exc)

    def _rerank(
        self,
        query: str,
        fused: List[Tuple[int, float]],
        top_k: int,
    ) -> List[SearchResult]:
        """
        Score the top candidates from RRF fusion using the configured re-ranker
        and return final SearchResult dicts.

        Falls back to RRF scores if the re-ranker is unavailable.
        """
        if not fused:
            return []

        # Take up to max(top_k_dense, top_k_bm25) candidates into the re-ranker.
        max_candidates = max(self._cfg.top_k_dense, self._cfg.top_k_bm25)
        candidates: List[Tuple[int, float]] = fused[:max_candidates]

        # ── Cohere re-rank ────────────────────────────────────────────────
        if self._cohere_client is not None:
            return self._rerank_cohere(query, candidates, top_k)

        # ── Cross-encoder re-rank ─────────────────────────────────────────
        if self._cross_encoder is not None:
            return self._rerank_cross_encoder(query, candidates, top_k)

        # ── RRF scores only ───────────────────────────────────────────────
        return self._hits_to_results(candidates[:top_k])

    def _rerank_cross_encoder(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        top_k: int,
    ) -> List[SearchResult]:
        try:
            texts   = [self._corpus[idx]["content"] for idx, _ in candidates]
            pairs   = [(query, t) for t in texts]
            ce_scores: List[float] = self._cross_encoder.predict(pairs).tolist()

            # Sigmoid-normalise logits so they sit in (0, 1).
            # Clamp to avoid exact 0.0 / 1.0 from extreme logits (e.g. ±100).
            _EPS = 1e-7

            def _sigmoid(x: float) -> float:
                raw = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))
                return max(_EPS, min(1.0 - _EPS, raw))

            scored = [
                (candidates[i][0], _sigmoid(ce_scores[i]))
                for i in range(len(candidates))
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            return self._hits_to_results(scored[:top_k])
        except Exception as exc:
            logger.error("vector_store: CrossEncoder rerank failed — %s", exc)
            return self._hits_to_results(candidates[:top_k])

    def _rerank_cohere(
        self,
        query: str,
        candidates: List[Tuple[int, float]],
        top_k: int,
    ) -> List[SearchResult]:
        try:
            docs = [self._corpus[idx]["content"] for idx, _ in candidates]
            response = self._cohere_client.rerank(
                model=self._cfg.cohere_rerank_model,
                query=query,
                documents=docs,
                top_n=top_k,
            )
            results: List[SearchResult] = []
            for hit in response.results:
                idx = candidates[hit.index][0]
                results.append({
                    "content": self._corpus[idx]["content"],
                    "source":  self._corpus[idx]["source"],
                    "score":   round(float(hit.relevance_score), 4),
                })
            return results
        except Exception as exc:
            logger.error("vector_store: Cohere rerank failed — %s", exc)
            return self._hits_to_results(candidates[:top_k])

    def _hits_to_results(
        self, hits: List[Tuple[int, float]]
    ) -> List[SearchResult]:
        """Convert (corpus_index, score) pairs into SearchResult dicts."""
        results = []
        for idx, score in hits:
            if 0 <= idx < len(self._corpus):
                results.append({
                    "content": self._corpus[idx]["content"],
                    "source":  self._corpus[idx]["source"],
                    "score":   round(float(score), 6),
                })
        return results

    # ------------------------------------------------------------------
    # Introspection helpers (useful in tests / observability)
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of chunks currently indexed."""
        return len(self._corpus)

    @property
    def has_dense(self) -> bool:
        """True if Qdrant dense index is available."""
        return self._qdrant is not None and self._embedder is not None

    @property
    def has_sparse(self) -> bool:
        """True if BM25 index is available."""
        return self._bm25_index is not None

    @property
    def has_reranker(self) -> bool:
        """True if a re-ranker (cross-encoder or Cohere) is loaded."""
        return self._cross_encoder is not None or self._cohere_client is not None


# ---------------------------------------------------------------------------
# Module-level singleton  (index built lazily on first search)
# ---------------------------------------------------------------------------

def _default_settings() -> VectorStoreSettings:
    """Build default settings resolving paths relative to the repo root."""
    _repo_root = Path(__file__).parent.parent
    return VectorStoreSettings(
        docs_dir=_repo_root / "data" / "policy_docs",
        qdrant_local_path=_repo_root / "data" / "qdrant_db",
        qdrant_mode=QdrantMode.local,
        reranker_backend=RerankerBackend.cross_encoder,
    )


policy_store = PolicyVectorStore(settings=_default_settings())
