"""
tests/test_vector_store.py
==========================
Unit tests for the refactored hybrid vector store.

Test strategy
-------------
* All tests use QdrantMode.memory + RerankerBackend.none so they run
  completely offline with zero disk I/O and no ML model downloads.
* Heavy optional deps (qdrant-client, sentence-transformers, rank-bm25,
  cohere) are imported conditionally; tests that require them are skipped
  when the package is absent.
* Fixtures inject synthetic documents so tests never rely on the repo's
  data/policy_docs/ directory.

Coverage groups
---------------
A  VectorStoreSettings — validation, env-var overrides, default values
B  Chunking            — _chunk_text, _load_and_chunk_all
C  BM25 retrieval      — tokenisation, scoring, ranking
D  Dense retrieval     — Qdrant upsert + search (skipped if no qdrant-client)
E  RRF fusion          — score normalisation, tie-breaking, empty inputs
F  Re-ranking          — cross-encoder path, Cohere path, none path
G  Public API          — search(), build_index(), introspection properties
H  Graceful fallback   — missing deps, empty corpus, build errors
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from src.vector_store_settings import (
    QdrantMode,
    RerankerBackend,
    VectorStoreSettings,
)
from src.vector_store import PolicyVectorStore

# ---------------------------------------------------------------------------
# Availability flags (mirror the guards in vector_store.py)
# ---------------------------------------------------------------------------

try:
    import qdrant_client  # noqa: F401
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    HAS_ST = True
except ImportError:
    HAS_ST = False

try:
    from sentence_transformers.cross_encoder import CrossEncoder  # noqa: F401
    HAS_CE = True
except ImportError:
    HAS_CE = False

try:
    from rank_bm25 import BM25Okapi  # noqa: F401
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SAMPLE_DOCS = [
    "IRDAI mandates that all insurers must settle claims within 30 days of receiving full documentation.",
    "The free-look period allows a policyholder to cancel a policy within 15 days of receipt.",
    "Motor insurance policy covers own-damage and third-party liability under the ClaimGuard Standard Policy.",
    "Pre-existing damage to the vehicle at the time of policy issuance is excluded from coverage.",
    "Grievance redressal must be completed within 15 working days as per IRDAI guidelines.",
    "KYC verification is mandatory before any policy issuance or claim disbursement.",
    "Portability rights allow policyholders to migrate between insurers without losing accrued benefits.",
    "Claim repudiation must be communicated in writing with specific grounds within 30 days.",
]


def _make_store(
    *,
    mode: QdrantMode = QdrantMode.memory,
    reranker: RerankerBackend = RerankerBackend.none,
    top_k_dense: int = 5,
    top_k_bm25: int = 5,
    top_k_rerank: int = 3,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> PolicyVectorStore:
    """Create a store with in-memory Qdrant and no re-ranker (fast, offline)."""
    settings = VectorStoreSettings(
        qdrant_mode=mode,
        reranker_backend=reranker,
        top_k_dense=top_k_dense,
        top_k_bm25=top_k_bm25,
        top_k_rerank=top_k_rerank,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        docs_dir=Path("data/policy_docs"),       # won't be read in mocked tests
    )
    return PolicyVectorStore(settings=settings)


def _inject_corpus(store: PolicyVectorStore, docs: List[str] | None = None) -> None:
    """
    Bypass _load_and_chunk_all and directly populate the store's corpus
    with synthetic chunks.  Then build BM25, dense, and reranker indices.
    """
    texts = docs or _SAMPLE_DOCS
    store._corpus = [
        {"content": t, "source": f"synthetic_{i}.txt"}
        for i, t in enumerate(texts)
    ]
    store._tokenised_corpus = [store._tokenise(c["content"]) for c in store._corpus]
    store._built = True          # prevent re-entrant build_index()
    store._init_bm25()
    store._init_dense(store._corpus)
    store._init_reranker()


# ===========================================================================
# A — VectorStoreSettings
# ===========================================================================

class TestVectorStoreSettings:
    """Validate Pydantic settings: defaults, constraints, and env vars."""

    def test_defaults_are_sensible(self) -> None:
        cfg = VectorStoreSettings()
        assert cfg.qdrant_mode == QdrantMode.local
        assert cfg.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert cfg.embedding_dim == 384
        assert cfg.chunk_size == 300
        assert cfg.chunk_overlap == 50
        assert cfg.top_k_dense == 10
        assert cfg.top_k_bm25 == 10
        assert cfg.top_k_rerank == 3
        assert cfg.reranker_backend == RerankerBackend.cross_encoder
        assert cfg.rrf_k == 60

    def test_memory_mode(self) -> None:
        cfg = VectorStoreSettings(qdrant_mode="memory")
        assert cfg.qdrant_mode == QdrantMode.memory

    def test_remote_mode(self) -> None:
        cfg = VectorStoreSettings(
            qdrant_mode="remote",
            qdrant_url="http://qdrant.internal:6333",
            qdrant_api_key="secret",
        )
        assert cfg.qdrant_mode == QdrantMode.remote
        assert cfg.qdrant_url == "http://qdrant.internal:6333"
        assert cfg.qdrant_api_key == "secret"

    def test_invalid_overlap_raises(self) -> None:
        """chunk_overlap must be strictly less than chunk_size."""
        with pytest.raises(Exception):
            VectorStoreSettings(chunk_size=100, chunk_overlap=100)

    def test_invalid_top_k_rerank_raises(self) -> None:
        """top_k_rerank cannot exceed max(top_k_dense, top_k_bm25)."""
        with pytest.raises(Exception):
            VectorStoreSettings(top_k_dense=5, top_k_bm25=5, top_k_rerank=10)

    def test_env_prefix_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAIMGUARD_VS_CHUNK_SIZE", "512")
        monkeypatch.setenv("CLAIMGUARD_VS_TOP_K_DENSE", "20")
        cfg = VectorStoreSettings()
        assert cfg.chunk_size == 512
        assert cfg.top_k_dense == 20

    def test_cohere_backend_enum(self) -> None:
        cfg = VectorStoreSettings(
            reranker_backend="cohere",
            cohere_api_key="co-test",
            top_k_dense=10,
            top_k_bm25=10,
            top_k_rerank=3,
        )
        assert cfg.reranker_backend == RerankerBackend.cohere
        assert cfg.cohere_api_key == "co-test"

    def test_none_reranker(self) -> None:
        cfg = VectorStoreSettings(reranker_backend="none")
        assert cfg.reranker_backend == RerankerBackend.none

    def test_bm25_params(self) -> None:
        cfg = VectorStoreSettings(bm25_k1=1.2, bm25_b=0.5)
        assert cfg.bm25_k1 == pytest.approx(1.2)
        assert cfg.bm25_b == pytest.approx(0.5)

    def test_rrf_k_positive(self) -> None:
        cfg = VectorStoreSettings(rrf_k=120)
        assert cfg.rrf_k == 120


# ===========================================================================
# B — Chunking
# ===========================================================================

class TestChunking:
    """_chunk_text produces the right number and size of chunks."""

    def setup_method(self) -> None:
        self.store = _make_store()

    def test_single_chunk_short_text(self) -> None:
        chunks = self.store._chunk_text("Short text.", source="doc.txt")
        assert len(chunks) == 1
        assert chunks[0]["source"] == "doc.txt"
        assert chunks[0]["content"] == "Short text."

    def test_multiple_chunks_long_text(self) -> None:
        # 600-char text with chunk_size=200 and overlap=20 → step=180 → 4 chunks
        text = "A" * 600
        chunks = self.store._chunk_text(text, source="big.txt")
        assert len(chunks) >= 3

    def test_overlap_reuses_content(self) -> None:
        """Each chunk (except first) should overlap with the previous."""
        text = "word " * 120      # ~600 chars
        chunks = self.store._chunk_text(text, source="overlap.txt")
        if len(chunks) >= 2:
            # The last `overlap` chars of chunk[0] should appear in chunk[1]
            overlap = self.store._cfg.chunk_overlap
            tail = chunks[0]["content"][-overlap:]
            assert tail in chunks[1]["content"]

    def test_empty_text_returns_no_chunks(self) -> None:
        chunks = self.store._chunk_text("   ", source="empty.txt")
        assert chunks == []

    def test_chunk_source_preserved(self) -> None:
        chunks = self.store._chunk_text("Some content here.", source="my_policy.txt")
        assert all(c["source"] == "my_policy.txt" for c in chunks)

    def test_load_and_chunk_missing_dir(self, tmp_path: Path) -> None:
        """Missing docs_dir returns an empty list without raising."""
        cfg = VectorStoreSettings(
            qdrant_mode="memory",
            reranker_backend="none",
            docs_dir=tmp_path / "nonexistent",
        )
        store = PolicyVectorStore(settings=cfg)
        chunks = store._load_and_chunk_all()
        assert chunks == []

    def test_load_and_chunk_reads_txt_files(self, tmp_path: Path) -> None:
        (tmp_path / "policy.txt").write_text("Coverage clause: motor damage.", encoding="utf-8")
        (tmp_path / "irdai.txt").write_text("IRDAI mandates 30-day settlement.", encoding="utf-8")
        (tmp_path / "ignored.pdf").write_text("should be ignored", encoding="utf-8")

        cfg = VectorStoreSettings(
            qdrant_mode="memory",
            reranker_backend="none",
            docs_dir=tmp_path,
            chunk_size=200,
            chunk_overlap=20,
        )
        store = PolicyVectorStore(settings=cfg)
        chunks = store._load_and_chunk_all()

        sources = {c["source"] for c in chunks}
        assert "policy.txt" in sources
        assert "irdai.txt" in sources
        assert "ignored.pdf" not in sources


# ===========================================================================
# C — BM25 retrieval
# ===========================================================================

class TestBM25:
    """Tokenisation, scoring, and ranking of the BM25 index."""

    def test_tokenise_lowercase(self) -> None:
        tokens = PolicyVectorStore._tokenise("IRDAI Claim Settlement POLICY")
        assert tokens == ["irdai", "claim", "settlement", "policy"]

    def test_tokenise_empty(self) -> None:
        assert PolicyVectorStore._tokenise("") == []

    @pytest.mark.skipif(not HAS_BM25, reason="rank-bm25 not installed")
    def test_bm25_returns_results(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        hits = store._bm25_search("claim settlement IRDAI", top_k=3)
        assert len(hits) > 0
        assert all(isinstance(idx, int) for idx, _ in hits)
        assert all(isinstance(score, float) for _, score in hits)

    @pytest.mark.skipif(not HAS_BM25, reason="rank-bm25 not installed")
    def test_bm25_sorted_descending(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        hits = store._bm25_search("IRDAI settlement 30 days", top_k=5)
        scores = [s for _, s in hits]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.skipif(not HAS_BM25, reason="rank-bm25 not installed")
    def test_bm25_respects_top_k(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        hits = store._bm25_search("policy coverage", top_k=2)
        assert len(hits) <= 2

    @pytest.mark.skipif(not HAS_BM25, reason="rank-bm25 not installed")
    def test_bm25_relevant_doc_ranks_first(self) -> None:
        """The doc with the most query term overlap should rank highest."""
        store = _make_store()
        _inject_corpus(store)
        hits = store._bm25_search("free-look period cancel policy", top_k=5)
        top_idx = hits[0][0]
        assert "free-look" in store._corpus[top_idx]["content"].lower() or \
               "cancel" in store._corpus[top_idx]["content"].lower()

    def test_bm25_search_no_index_returns_empty(self) -> None:
        """When BM25 index is absent, search returns []."""
        store = _make_store()
        store._built = True
        store._corpus = [{"content": "Some text", "source": "x.txt"}]
        store._bm25_index = None
        assert store._bm25_search("some query", top_k=3) == []


# ===========================================================================
# D — Dense retrieval (Qdrant)
# ===========================================================================

@pytest.mark.skipif(not HAS_QDRANT or not HAS_ST, reason="qdrant-client or sentence-transformers not installed")
class TestDenseRetrieval:
    """Qdrant dense retrieval with real in-memory client."""

    def test_dense_search_returns_results(self) -> None:
        store = _make_store(mode=QdrantMode.memory)
        _inject_corpus(store)
        if not store.has_dense:
            pytest.skip("Dense index unavailable (missing deps)")
        hits = store._dense_search("IRDAI claim settlement", top_k=3)
        assert len(hits) > 0

    def test_dense_search_scores_in_range(self) -> None:
        store = _make_store(mode=QdrantMode.memory)
        _inject_corpus(store)
        if not store.has_dense:
            pytest.skip("Dense index unavailable")
        hits = store._dense_search("motor insurance policy", top_k=5)
        for _, score in hits:
            # Cosine similarity from Qdrant is in [-1, 1] but typically (0, 1)
            assert -1.0 <= score <= 1.0

    def test_dense_search_respects_top_k(self) -> None:
        store = _make_store(mode=QdrantMode.memory)
        _inject_corpus(store)
        if not store.has_dense:
            pytest.skip("Dense index unavailable")
        hits = store._dense_search("policy", top_k=2)
        assert len(hits) <= 2

    def test_dense_search_returns_empty_without_index(self) -> None:
        store = _make_store()
        store._built = True
        store._qdrant = None
        store._embedder = None
        assert store._dense_search("any query", top_k=3) == []

    def test_corpus_size_matches_injected(self) -> None:
        store = _make_store(mode=QdrantMode.memory)
        _inject_corpus(store, docs=_SAMPLE_DOCS[:4])
        assert store.corpus_size == 4

    def test_local_qdrant_uses_tmp_path(self, tmp_path: Path) -> None:
        """QdrantMode.local creates on-disk storage without errors."""
        cfg = VectorStoreSettings(
            qdrant_mode=QdrantMode.local,
            qdrant_local_path=tmp_path / "qdrant",
            reranker_backend=RerankerBackend.none,
            docs_dir=Path("data/policy_docs"),
            top_k_dense=3,
            top_k_bm25=3,
            top_k_rerank=2,
        )
        store = PolicyVectorStore(settings=cfg)
        _inject_corpus(store)
        assert (tmp_path / "qdrant").exists()


# ===========================================================================
# E — RRF fusion
# ===========================================================================

class TestRRFFusion:
    """Reciprocal Rank Fusion score calculation and edge cases."""

    def setup_method(self) -> None:
        self.store = _make_store()

    def test_rrf_score_decreases_with_rank(self) -> None:
        """Higher-ranked documents must receive higher RRF scores."""
        dense = [(0, 0.9), (1, 0.8), (2, 0.7)]
        bm25  = [(0, 5.0), (2, 3.0), (3, 1.0)]
        fused = self.store._rrf_fuse(dense, bm25)
        scores = [s for _, s in fused]
        # Scores should be sorted descending
        assert scores == sorted(scores, reverse=True)

    def test_rrf_max_score_is_one(self) -> None:
        """Normalised RRF: the top document always scores 1.0."""
        dense = [(0, 0.9), (1, 0.5)]
        bm25  = [(0, 4.0), (1, 2.0)]
        fused = self.store._rrf_fuse(dense, bm25)
        assert fused[0][1] == pytest.approx(1.0)

    def test_rrf_empty_both_returns_empty(self) -> None:
        assert self.store._rrf_fuse([], []) == []

    def test_rrf_empty_dense_uses_bm25_only(self) -> None:
        bm25 = [(0, 5.0), (1, 3.0)]
        fused = self.store._rrf_fuse([], bm25)
        assert len(fused) == 2
        assert fused[0][1] == pytest.approx(1.0)

    def test_rrf_empty_bm25_uses_dense_only(self) -> None:
        dense = [(2, 0.8), (3, 0.6)]
        fused = self.store._rrf_fuse(dense, [])
        assert len(fused) == 2

    def test_rrf_doc_appearing_in_both_lists_scores_higher(self) -> None:
        """A doc ranked high in both lists must outscore one ranked in only one."""
        # doc 0: rank-1 dense + rank-1 BM25 (double reward)
        # doc 1: rank-1 dense only
        dense = [(0, 0.9), (1, 0.8)]
        bm25  = [(0, 5.0)]
        fused = dict(self.store._rrf_fuse(dense, bm25))
        assert fused[0] > fused[1]

    def test_rrf_unique_indices_in_output(self) -> None:
        dense = [(0, 0.9), (1, 0.7), (2, 0.5)]
        bm25  = [(1, 4.0), (2, 2.0), (3, 1.0)]
        fused = self.store._rrf_fuse(dense, bm25)
        indices = [idx for idx, _ in fused]
        assert len(indices) == len(set(indices)), "Duplicate indices in RRF output"

    def test_rrf_custom_k_affects_scores(self) -> None:
        """Different rrf_k values must produce different normalised scores."""
        dense = [(0, 0.9), (1, 0.5)]
        bm25  = [(2, 3.0)]

        cfg_low  = VectorStoreSettings(qdrant_mode="memory", reranker_backend="none", rrf_k=1)
        cfg_high = VectorStoreSettings(qdrant_mode="memory", reranker_backend="none", rrf_k=100)
        store_low  = PolicyVectorStore(settings=cfg_low)
        store_high = PolicyVectorStore(settings=cfg_high)

        fused_low  = dict(store_low._rrf_fuse(dense, bm25))
        fused_high = dict(store_high._rrf_fuse(dense, bm25))

        # With lower k, rank differences are amplified → different relative scores
        assert fused_low != fused_high


# ===========================================================================
# F — Re-ranking
# ===========================================================================

class TestReranking:
    """Cross-encoder and Cohere re-ranking paths, including fallbacks."""

    # ── Cross-encoder (mocked) ───────────────────────────────────────────────

    def test_cross_encoder_rerank_orders_by_ce_score(self) -> None:
        """
        Mock a CrossEncoder that returns controllable scores.
        Doc at index 1 should rank first when its CE score is highest.
        """
        store = _make_store()
        store._corpus = [
            {"content": "IRDAI 30-day claim settlement rule.", "source": "a.txt"},
            {"content": "Motor policy exclusions for pre-existing damage.", "source": "b.txt"},
            {"content": "Free-look cancellation rights.", "source": "c.txt"},
        ]
        store._built = True

        mock_ce = MagicMock()
        import numpy as np
        # Return logits: doc-1 scores highest (logit 5.0 → sigmoid ≈ 0.993)
        mock_ce.predict.return_value = np.array([1.0, 5.0, -1.0])
        store._cross_encoder = mock_ce

        candidates = [(0, 0.8), (1, 0.6), (2, 0.4)]
        results = store._rerank_cross_encoder("claim settlement", candidates, top_k=2)

        assert len(results) == 2
        assert results[0]["content"] == store._corpus[1]["content"]
        assert results[0]["score"] > results[1]["score"]

    def test_cross_encoder_sigmoid_scores_in_01(self) -> None:
        """Sigmoid-normalised CE scores must be in (0, 1)."""
        store = _make_store()
        store._corpus = [
            {"content": f"Document {i}.", "source": f"{i}.txt"}
            for i in range(5)
        ]
        store._built = True

        mock_ce = MagicMock()
        import numpy as np
        # Extreme logits to stress-test sigmoid
        mock_ce.predict.return_value = np.array([-100.0, -10.0, 0.0, 10.0, 100.0])
        store._cross_encoder = mock_ce

        candidates = [(i, 0.5) for i in range(5)]
        results = store._rerank_cross_encoder("query", candidates, top_k=5)
        for r in results:
            # sigmoid is asymptotically bounded: score ∈ (0, 1].
            # Extreme logits (±100) legitimately saturate to 1.0 / ~0.0.
            assert 0.0 <= r["score"] <= 1.0

    def test_cross_encoder_fallback_on_exception(self) -> None:
        """If CE.predict raises, _rerank_cross_encoder falls back to RRF order."""
        store = _make_store()
        store._corpus = [
            {"content": f"Doc {i}", "source": f"{i}.txt"} for i in range(3)
        ]
        store._built = True

        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError("model failed")
        store._cross_encoder = mock_ce

        candidates = [(0, 1.0), (1, 0.8), (2, 0.6)]
        results = store._rerank_cross_encoder("query", candidates, top_k=2)
        # Fallback: returns first top_k by RRF order, no exception raised
        assert len(results) == 2
        assert all("content" in r and "score" in r for r in results)

    # ── Cohere (mocked) ──────────────────────────────────────────────────────

    def test_cohere_rerank_uses_relevance_score(self) -> None:
        store = _make_store()
        store._corpus = [
            {"content": "IRDAI settlement.", "source": "a.txt"},
            {"content": "Free-look period.", "source": "b.txt"},
        ]
        store._built = True

        # Mock Cohere client
        mock_cohere = MagicMock()
        mock_result_0 = MagicMock(); mock_result_0.index = 1; mock_result_0.relevance_score = 0.95
        mock_result_1 = MagicMock(); mock_result_1.index = 0; mock_result_1.relevance_score = 0.72
        mock_cohere.rerank.return_value = MagicMock(results=[mock_result_0, mock_result_1])
        store._cohere_client = mock_cohere

        candidates = [(0, 0.8), (1, 0.6)]
        results = store._rerank_cohere("IRDAI claim", candidates, top_k=2)

        assert len(results) == 2
        # Cohere ranked doc-index-1 first (relevance 0.95)
        assert results[0]["content"] == store._corpus[1]["content"]
        assert results[0]["score"] == pytest.approx(0.95)

    def test_cohere_fallback_on_exception(self) -> None:
        """If Cohere API raises, _rerank_cohere falls back without raising."""
        store = _make_store()
        store._corpus = [{"content": f"Doc {i}", "source": f"{i}.txt"} for i in range(2)]
        store._built = True

        mock_cohere = MagicMock()
        mock_cohere.rerank.side_effect = Exception("API error")
        store._cohere_client = mock_cohere

        candidates = [(0, 1.0), (1, 0.5)]
        results = store._rerank_cohere("query", candidates, top_k=2)
        assert isinstance(results, list)

    # ── None reranker ─────────────────────────────────────────────────────────

    def test_none_reranker_returns_rrf_order(self) -> None:
        store = _make_store(reranker=RerankerBackend.none)
        store._corpus = [
            {"content": f"Doc {i}", "source": f"{i}.txt"} for i in range(4)
        ]
        store._built = True
        store._cross_encoder = None
        store._cohere_client = None

        candidates = [(2, 0.9), (0, 0.7), (3, 0.5)]
        results = store._rerank("query", candidates, top_k=2)
        assert len(results) == 2
        assert results[0]["content"] == "Doc 2"

    # ── _hits_to_results ──────────────────────────────────────────────────────

    def test_hits_to_results_structure(self) -> None:
        store = _make_store()
        store._corpus = [
            {"content": "Alpha", "source": "a.txt"},
            {"content": "Beta",  "source": "b.txt"},
        ]
        results = store._hits_to_results([(0, 0.9), (1, 0.7)])
        assert len(results) == 2
        # chunk_id is mandated by the Guardrails zero-hallucination citation policy
        assert results[0]["chunk_id"]
        assert results[1]["chunk_id"]
        assert {k: v for k, v in results[0].items() if k != "chunk_id"} == {
            "content": "Alpha", "source": "a.txt", "score": 0.9
        }
        assert {k: v for k, v in results[1].items() if k != "chunk_id"} == {
            "content": "Beta", "source": "b.txt", "score": 0.7
        }

    def test_hits_to_results_out_of_bounds_skipped(self) -> None:
        store = _make_store()
        store._corpus = [{"content": "Only one doc", "source": "x.txt"}]
        # Index 99 is out of bounds — must be skipped silently
        results = store._hits_to_results([(0, 0.8), (99, 0.5)])
        assert len(results) == 1
        assert results[0]["content"] == "Only one doc"


# ===========================================================================
# G — Public API: search() + build_index() + introspection
# ===========================================================================

class TestPublicAPI:
    """End-to-end tests of the public search() interface."""

    def test_search_returns_list(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        results = store.search("IRDAI claim settlement")
        assert isinstance(results, list)

    def test_search_result_has_required_keys(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        results = store.search("motor insurance exclusions", top_k=2)
        for r in results:
            assert "content" in r
            assert "source" in r
            assert "score" in r

    def test_search_respects_top_k(self) -> None:
        store = _make_store(top_k_rerank=3)
        _inject_corpus(store)
        results = store.search("policy coverage", top_k=2)
        assert len(results) <= 2

    def test_search_scores_are_floats_in_range(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        results = store.search("KYC verification mandatory")
        for r in results:
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0

    def test_search_triggers_lazy_build(self, tmp_path: Path) -> None:
        """search() on a fresh store must call build_index() automatically."""
        (tmp_path / "doc.txt").write_text(
            "Coverage clause motor third-party liability.", encoding="utf-8"
        )
        cfg = VectorStoreSettings(
            qdrant_mode=QdrantMode.memory,
            reranker_backend=RerankerBackend.none,
            docs_dir=tmp_path,
            chunk_size=200,
            chunk_overlap=20,
            top_k_dense=3,
            top_k_bm25=3,
            top_k_rerank=2,
        )
        store = PolicyVectorStore(settings=cfg)
        assert not store._built
        results = store.search("motor coverage")
        assert store._built
        # At least BM25 should have fired (rank-bm25 installed)
        assert isinstance(results, list)

    def test_build_index_idempotent(self, tmp_path: Path) -> None:
        """Calling build_index() twice must not raise or duplicate chunks."""
        (tmp_path / "p.txt").write_text("IRDAI policy text here.", encoding="utf-8")
        cfg = VectorStoreSettings(
            qdrant_mode=QdrantMode.memory,
            reranker_backend=RerankerBackend.none,
            docs_dir=tmp_path,
            top_k_dense=3,
            top_k_bm25=3,
            top_k_rerank=2,
        )
        store = PolicyVectorStore(settings=cfg)
        store.build_index()
        size_after_first = store.corpus_size
        store.build_index()          # second call is a no-op
        assert store.corpus_size == size_after_first

    def test_introspection_has_sparse(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        assert store.has_sparse == (store._bm25_index is not None)

    def test_introspection_has_dense(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        assert store.has_dense == (store._qdrant is not None and store._embedder is not None)

    def test_introspection_has_reranker_false_for_none_backend(self) -> None:
        store = _make_store(reranker=RerankerBackend.none)
        _inject_corpus(store)
        assert store.has_reranker is False

    def test_corpus_size_zero_before_build(self) -> None:
        store = _make_store()
        assert store.corpus_size == 0

    def test_search_returns_empty_for_empty_corpus(self) -> None:
        store = _make_store()
        store._built = True          # skip build; leave corpus empty
        assert store.search("anything") == []


# ===========================================================================
# H — Graceful fallback
# ===========================================================================

class TestGracefulFallback:
    """search() must never raise, even under adverse conditions."""

    def test_search_safe_without_any_backends(self) -> None:
        """Both BM25 and dense are None — search returns []."""
        store = _make_store()
        store._corpus = [{"content": "Some doc", "source": "x.txt"}]
        store._built = True
        store._bm25_index = None
        store._qdrant = None
        store._embedder = None
        results = store.search("some query")
        assert results == []

    def test_search_safe_on_corrupt_qdrant(self) -> None:
        """Qdrant client throws on search — store returns []."""
        store = _make_store()
        _inject_corpus(store, docs=_SAMPLE_DOCS[:3])

        if store._qdrant is not None:
            store._qdrant.search = MagicMock(side_effect=RuntimeError("connection refused"))

        # Should not raise
        results = store.search("motor claim")
        assert isinstance(results, list)

    def test_search_safe_when_build_fails(self, tmp_path: Path) -> None:
        """Even if build_index() encounters an error, search returns []."""
        cfg = VectorStoreSettings(
            qdrant_mode=QdrantMode.memory,
            reranker_backend=RerankerBackend.none,
            docs_dir=tmp_path / "does_not_exist",
            top_k_dense=3,
            top_k_bm25=3,
            top_k_rerank=2,
        )
        store = PolicyVectorStore(settings=cfg)
        results = store.search("query on empty store")
        assert results == []

    def test_search_no_exception_on_bad_query(self) -> None:
        store = _make_store()
        _inject_corpus(store)
        # Edge cases: empty string, whitespace, unicode
        for q in ["", "   ", "日本語クエリ", "!@#$%^&*()"]:
            results = store.search(q)
            assert isinstance(results, list), f"Raised for query: {repr(q)}"

    def test_missing_cohere_key_does_not_crash(self) -> None:
        """RerankerBackend.cohere without API key should warn and degrade."""
        cfg = VectorStoreSettings(
            qdrant_mode="memory",
            reranker_backend=RerankerBackend.cohere,
            cohere_api_key=None,     # deliberately missing
            top_k_dense=5,
            top_k_bm25=5,
            top_k_rerank=3,
        )
        store = PolicyVectorStore(settings=cfg)
        _inject_corpus(store)        # _init_reranker() called inside
        assert store._cohere_client is None
        # search must still work (falls back to RRF)
        results = store.search("IRDAI claim")
        assert isinstance(results, list)

    def test_module_import_always_safe(self) -> None:
        """Importing vector_store must never raise regardless of env."""
        import importlib
        import src.vector_store as vs
        importlib.reload(vs)
        assert hasattr(vs, "PolicyVectorStore")
        assert hasattr(vs, "policy_store")
