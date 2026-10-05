"""
DatabaseManager for ClaimGuard AI.

Fallback chain for reads and writes:
  1. asyncpg (PostgreSQL)  — preferred when DATABASE_URL is set
  2. supabase-py           — when SUPABASE_URL + SUPABASE_KEY are set
  3. CSV / JSONL files     — always available, no credentials required

All methods are async.  CSV/JSONL access is offloaded to a thread via
``asyncio.to_thread`` so the event loop is never blocked.

Environment variables
---------------------
DATABASE_URL   — asyncpg-compatible PostgreSQL DSN
SUPABASE_URL   — Supabase project URL
SUPABASE_KEY   — Supabase anon/service key
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_CLAIMS_CSV = _DATA_DIR / "sample_claims.csv"
_POLICIES_CSV = _DATA_DIR / "sample_policies.csv"
_DECISIONS_JSONL = _DATA_DIR / "decisions.jsonl"

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
try:
    import asyncpg  # type: ignore

    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    from supabase import create_client as _supabase_create_client  # type: ignore

    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

try:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_filters(records: List[Dict], filters: Optional[Dict]) -> List[Dict]:
    """Filter a list of dicts by equality on each key/value in *filters*."""
    if not filters:
        return records
    return [
        r for r in records if all(str(r.get(k)) == str(v) for k, v in filters.items())
    ]


def _csv_to_records(path: Path, filters: Optional[Dict]) -> List[Dict]:
    """Read *path* as CSV and return filtered records (runs in a thread)."""
    if not HAS_PANDAS:
        logger.warning("pandas not available; returning empty list for %s", path)
        return []
    if not path.exists():
        logger.warning("CSV file not found: %s", path)
        return []
    df = pd.read_csv(path)
    records: List[Dict] = df.to_dict(orient="records")
    return _apply_filters(records, filters)


def _append_jsonl(path: Path, record: Dict) -> None:
    """Append *record* as a JSON line to *path* (runs in a thread)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """Async data-access layer with a three-tier fallback chain.

    No connection is opened at construction time.  Each method establishes
    (and closes) its connection independently so the class is safe to
    instantiate at module import.
    """

    def __init__(self) -> None:
        self._database_url: Optional[str] = os.environ.get("DATABASE_URL")
        self._supabase_url: Optional[str] = os.environ.get("SUPABASE_URL")
        self._supabase_key: Optional[str] = os.environ.get("SUPABASE_KEY")

    # ------------------------------------------------------------------
    # Internal: asyncpg helpers
    # ------------------------------------------------------------------

    async def _asyncpg_fetch(self, query: str, *args: Any) -> List[Dict]:
        """Execute *query* and return rows as dicts via asyncpg."""
        conn = await asyncpg.connect(self._database_url)
        try:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def _asyncpg_execute(self, query: str, *args: Any) -> None:
        """Execute a DML statement via asyncpg."""
        conn = await asyncpg.connect(self._database_url)
        try:
            await conn.execute(query, *args)
        finally:
            await conn.close()

    # ------------------------------------------------------------------
    # Internal: supabase helpers
    # ------------------------------------------------------------------

    def _get_supabase_client(self):  # type: ignore[return]
        return _supabase_create_client(self._supabase_url, self._supabase_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_claims(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Return claim records matching *filters*.

        Falls through asyncpg → supabase → CSV on each failure.
        """
        # Tier 1: asyncpg
        if HAS_ASYNCPG and self._database_url:
            try:
                rows = await self._asyncpg_fetch("SELECT * FROM claims LIMIT 1000")
                return _apply_filters(rows, filters)
            except Exception as exc:
                logger.warning("asyncpg get_claims failed (%s); trying supabase.", exc)

        # Tier 2: supabase
        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                query = client.table("claims").select("*")
                if filters:
                    for k, v in filters.items():
                        query = query.eq(k, v)
                result = query.execute()
                return result.data or []
            except Exception as exc:
                logger.warning("supabase get_claims failed (%s); falling back to CSV.", exc)

        # Tier 3: CSV
        return await asyncio.to_thread(_csv_to_records, _CLAIMS_CSV, filters)

    async def get_policies(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Return policy records matching *filters*.

        Falls through asyncpg → supabase → CSV on each failure.
        """
        # Tier 1: asyncpg
        if HAS_ASYNCPG and self._database_url:
            try:
                rows = await self._asyncpg_fetch("SELECT * FROM policies LIMIT 1000")
                return _apply_filters(rows, filters)
            except Exception as exc:
                logger.warning("asyncpg get_policies failed (%s); trying supabase.", exc)

        # Tier 2: supabase
        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                query = client.table("policies").select("*")
                if filters:
                    for k, v in filters.items():
                        query = query.eq(k, v)
                result = query.execute()
                return result.data or []
            except Exception as exc:
                logger.warning(
                    "supabase get_policies failed (%s); falling back to CSV.", exc
                )

        # Tier 3: CSV
        return await asyncio.to_thread(_csv_to_records, _POLICIES_CSV, filters)

    async def save_decision(self, decision: Dict) -> None:
        """Persist a decision record.

        Falls through asyncpg → supabase → JSONL append on each failure.
        """
        # Tier 1: asyncpg
        if HAS_ASYNCPG and self._database_url:
            try:
                columns = ", ".join(decision.keys())
                placeholders = ", ".join(f"${i + 1}" for i in range(len(decision)))
                query = f"INSERT INTO decisions ({columns}) VALUES ({placeholders})"
                await self._asyncpg_execute(query, *decision.values())
                return
            except Exception as exc:
                logger.warning("asyncpg save_decision failed (%s); trying supabase.", exc)

        # Tier 2: supabase
        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                client.table("decisions").insert(decision).execute()
                return
            except Exception as exc:
                logger.warning(
                    "supabase save_decision failed (%s); falling back to JSONL.", exc
                )

        # Tier 3: JSONL append
        await asyncio.to_thread(_append_jsonl, _DECISIONS_JSONL, decision)
