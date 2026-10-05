"""
Policy document vector store for ClaimGuard AI Policy Copilot.

Reads .txt files from data/policy_docs/, chunks them with a sliding window,
embeds them using sentence-transformers (all-MiniLM-L6-v2, local, no API key),
and persists a ChromaDB collection at data/chroma_db/.

Graceful fallback: if chromadb or sentence_transformers are absent, search()
returns an empty list and the agent pipeline continues without retrieval.

The index is built lazily — build_index() is called on the first search(), not
at import time, so importing this module is always safe.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import chromadb  # type: ignore

    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore

    HAS_ST = True
except ImportError:
    HAS_ST = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 300      # characters per chunk
_CHUNK_OVERLAP = 50   # overlap between consecutive chunks
_COLLECTION_NAME = "policy_docs"


# ---------------------------------------------------------------------------
# Main vector store class
# ---------------------------------------------------------------------------


class PolicyVectorStore:
    """
    Semantic search over policy wording and IRDAI regulatory text.

    Usage
    -----
    store = PolicyVectorStore()
    results = store.search("coverage exclusion motor")
    """

    def __init__(
        self,
        docs_dir: Path = Path("data/policy_docs"),
        persist_dir: Path = Path("data/chroma_db"),
    ) -> None:
        self._docs_dir = docs_dir
        self._persist_dir = persist_dir
        self._built = False
        self._collection = None  # ChromaDB collection
        self._embedder = None    # SentenceTransformer model

    # ------------------------------------------------------------------
    # Chunking helper
    # ------------------------------------------------------------------

    def _load_and_chunk(self, text: str, source: str) -> List[dict]:
        """
        Split *text* into overlapping chunks of ~_CHUNK_SIZE characters.

        Returns
        -------
        list of dicts: [{content: str, source: str}, ...]
        """
        chunks: List[dict] = []
        start = 0
        while start < len(text):
            end = start + _CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"content": chunk, "source": source})
            start += _CHUNK_SIZE - _CHUNK_OVERLAP
        return chunks

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self) -> None:
        """
        Read all .txt files from docs_dir, chunk them, embed with
        all-MiniLM-L6-v2, and upsert into a ChromaDB collection.

        If ChromaDB or sentence-transformers are not installed, this method
        returns immediately — subsequent search() calls will return [].
        """
        if self._built:
            return

        self._built = True  # Mark early to prevent re-entrant calls.

        if not HAS_CHROMA or not HAS_ST:
            logger.warning(
                "vector_store: chromadb or sentence_transformers not installed; "
                "retrieval will return empty results."
            )
            return

        try:
            self._persist_dir.mkdir(parents=True, exist_ok=True)

            # Initialise ChromaDB persistent client.
            client = chromadb.PersistentClient(path=str(self._persist_dir))
            self._collection = client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

            # Load embedder.
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")

            # Read and chunk every .txt file.
            all_chunks: List[dict] = []
            for txt_file in sorted(self._docs_dir.glob("*.txt")):
                try:
                    text = txt_file.read_text(encoding="utf-8")
                    chunks = self._load_and_chunk(text, source=txt_file.name)
                    all_chunks.extend(chunks)
                except Exception as exc:
                    logger.warning("vector_store: could not read %s — %s", txt_file, exc)

            if not all_chunks:
                logger.warning("vector_store: no policy documents found in %s", self._docs_dir)
                return

            # Build IDs, texts, and metadata lists.
            ids = [f"chunk_{i}" for i in range(len(all_chunks))]
            texts = [c["content"] for c in all_chunks]
            metadatas = [{"source": c["source"]} for c in all_chunks]

            # Embed in one batch.
            embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

            # Upsert — safe to call on an existing collection (re-index).
            self._collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(
                "vector_store: indexed %d chunks from %s",
                len(all_chunks),
                self._docs_dir,
            )

        except Exception as exc:
            logger.error("vector_store: build_index failed — %s", exc)
            # Reset so search() degrades gracefully.
            self._collection = None
            self._embedder = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """
        Retrieve the top-k most relevant policy document chunks.

        Builds the index on first call (lazy initialisation).

        Returns
        -------
        list of dicts: [{content: str, source: str, score: float}, ...]
        Returns an empty list if the index is unavailable or if the query fails.
        """
        # Lazy build on first search.
        if not self._built:
            self.build_index()

        if not HAS_CHROMA or not HAS_ST:
            return []

        if self._collection is None or self._embedder is None:
            return []

        try:
            query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )

            hits: List[dict] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, dists):
                # ChromaDB cosine distance → similarity score.
                score = round(1.0 - float(dist), 4)
                hits.append(
                    {
                        "content": doc,
                        "source": meta.get("source", "unknown"),
                        "score": score,
                    }
                )

            return hits

        except Exception as exc:
            logger.error("vector_store: search failed — %s", exc)
            return []


# ---------------------------------------------------------------------------
# Module-level singleton  (index built lazily on first search)
# ---------------------------------------------------------------------------

policy_store = PolicyVectorStore(
    docs_dir=Path(__file__).parent.parent / "data" / "policy_docs",
    persist_dir=Path(__file__).parent.parent / "data" / "chroma_db",
)
