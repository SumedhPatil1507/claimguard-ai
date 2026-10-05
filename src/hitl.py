"""
Human-in-the-Loop (HITL) review queue for ClaimGuard AI.

Every AI-drafted underwriting or claims decision is enqueued here before any
downstream action is taken. Human analysts review, approve, reject, or escalate
each item. The queue is backed by an in-memory list persisted to
data/hitl_queue.json so items survive a process restart.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent / "data"
_QUEUE_FILE = _DATA_DIR / "hitl_queue.json"


# ---------------------------------------------------------------------------
# Pydantic v2 schemas
# ---------------------------------------------------------------------------


class HITLItem(BaseModel):
    """A single decision item awaiting human analyst review."""

    model_config = ConfigDict(str_strip_whitespace=True)

    item_id: str = Field(default_factory=lambda: uuid4().hex)
    session_id: str
    context_type: str  # e.g. 'underwriting' | 'claims'
    decision_draft: str
    model_result: dict
    analyst_review: Optional[str] = None
    status: Literal["pending", "approved", "rejected", "escalated"] = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# HITL Queue
# ---------------------------------------------------------------------------


class HITLQueue:
    """
    Thread-safe in-memory queue with JSON persistence.

    All writes (enqueue / review) acquire a threading.Lock so the queue is
    safe for concurrent FastAPI request handlers.
    """

    def __init__(self) -> None:
        self.items: List[HITLItem] = []
        self.lock: threading.Lock = threading.Lock()
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load persisted items from disk on startup (best-effort)."""
        try:
            if _QUEUE_FILE.exists():
                raw = json.loads(_QUEUE_FILE.read_text(encoding="utf-8"))
                self.items = [HITLItem.model_validate(entry) for entry in raw]
        except Exception:
            # Corrupt file or missing — start with an empty queue.
            self.items = []

    def _persist(self) -> None:
        """Write queue to disk (best-effort; called while lock is held)."""
        try:
            _DATA_DIR.mkdir(parents=True, exist_ok=True)
            payload = [item.model_dump(mode="json") for item in self.items]
            _QUEUE_FILE.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass  # Persistence failure must never crash the main pipeline.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enqueue(self, item: HITLItem) -> None:
        """Add a new decision item to the queue and persist to disk."""
        with self.lock:
            self.items.append(item)
            self._persist()

    def get_pending(self) -> List[HITLItem]:
        """Return all items whose status is 'pending'."""
        with self.lock:
            return [i for i in self.items if i.status == "pending"]

    def get_all(self) -> List[HITLItem]:
        """Return a snapshot of all items regardless of status."""
        with self.lock:
            return list(self.items)

    def review(
        self,
        item_id: str,
        decision: Literal["approved", "rejected", "escalated"],
        analyst_notes: str = "",
    ) -> Optional[HITLItem]:
        """
        Update a queue item with the analyst's decision.

        Parameters
        ----------
        item_id : hex string matching HITLItem.item_id
        decision : 'approved' | 'rejected' | 'escalated'
        analyst_notes : free-text review comment

        Returns
        -------
        The updated HITLItem, or None if item_id is not found.
        """
        with self.lock:
            for item in self.items:
                if item.item_id == item_id:
                    item.status = decision
                    item.analyst_review = analyst_notes
                    item.reviewed_at = datetime.utcnow()
                    self._persist()
                    return item
            return None

    def queue_depth(self) -> int:
        """Return the number of pending items."""
        with self.lock:
            return sum(1 for i in self.items if i.status == "pending")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

hitl_queue = HITLQueue()
