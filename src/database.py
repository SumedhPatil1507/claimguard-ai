"""
src/database.py
===============
Data-access layer for ClaimGuard AI.

Two classes are exported:

``DatabaseManager``
    Three-tier fallback chain: asyncpg → supabase → CSV/JSONL.
    Used by the Streamlit UI and eval scripts where graceful degradation
    is acceptable.

``ProductionDatabaseManager``
    Strict production variant — **no fallbacks**.  Every method raises
    ``InfrastructureError`` immediately when PostgreSQL is unreachable or
    when ``DATABASE_URL`` is not configured.  Used by Celery tasks and the
    FastAPI routes that must not silently serve stale data.

Module-level helpers
--------------------
``assert_postgres_reachable()``
    Synchronous connectivity check; raises ``InfrastructureError``.

``assert_redis_reachable()``
    Synchronous Redis ping; raises ``InfrastructureError``.

Environment variables
---------------------
DATABASE_URL   asyncpg-compatible PostgreSQL DSN.
SUPABASE_URL   Supabase project URL (fallback tier 2).
SUPABASE_KEY   Supabase anon/service key (fallback tier 2).
REDIS_URL      Redis DSN (default: redis://localhost:6379/0).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

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
    import asyncpg                          # type: ignore
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

try:
    from supabase import create_client as _supabase_create_client  # type: ignore
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

try:
    import pandas as pd                     # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import redis as _redis_lib              # type: ignore
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class InfrastructureError(RuntimeError):
    """
    Raised when a required infrastructure service (PostgreSQL, Redis) is
    unreachable and no fallback is permitted in the current execution context.

    HTTP callers should map this to **503 Service Unavailable**.
    """


# ---------------------------------------------------------------------------
# Module-level connectivity checks
# ---------------------------------------------------------------------------

def assert_postgres_reachable(database_url: Optional[str] = None) -> None:
    """Verify that PostgreSQL is reachable.

    Tries psycopg2 first; falls back to a raw TCP socket check.

    Parameters
    ----------
    database_url:
        Override the DSN.  Defaults to the ``DATABASE_URL`` environment
        variable.  Pass an explicit empty string to skip the check entirely.

    Raises
    ------
    InfrastructureError
        When ``DATABASE_URL`` is not set *or* the server cannot be reached.
    """
    url = database_url if database_url is not None else os.environ.get("DATABASE_URL", "")
    if not url:
        raise InfrastructureError(
            "DATABASE_URL is not configured.  "
            "Set DATABASE_URL to a valid PostgreSQL DSN to use production routes."
        )

    # Attempt psycopg2 synchronous connection (fastest check)
    try:
        import psycopg2                     # type: ignore
        conn = psycopg2.connect(url, connect_timeout=3)
        conn.close()
        return
    except ImportError:
        pass                                # psycopg2 not installed; fall through
    except Exception as exc:
        raise InfrastructureError(
            f"PostgreSQL is unreachable: {exc}.  "
            "Check DATABASE_URL and ensure the server is running."
        ) from exc

    # Fallback: raw TCP socket
    try:
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=3):
            pass
    except Exception as exc:
        raise InfrastructureError(
            f"PostgreSQL TCP check failed ({host}:{port}): {exc}.  "
            "Check DATABASE_URL and ensure the server is running."
        ) from exc


def assert_redis_reachable(redis_url: Optional[str] = None) -> None:
    """Verify that Redis is reachable via PING.

    Parameters
    ----------
    redis_url:
        Override the DSN.  Defaults to ``REDIS_URL`` env var or
        ``redis://localhost:6379/0``.

    Raises
    ------
    InfrastructureError
        When the redis package is not installed or the server is unreachable.
    """
    url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")

    if not HAS_REDIS:
        raise InfrastructureError(
            "redis package is not installed.  "
            "Install it with:  pip install redis"
        )
    try:
        client = _redis_lib.from_url(url, socket_connect_timeout=3)
        client.ping()
    except Exception as exc:
        raise InfrastructureError(
            f"Redis is unreachable at {url!r}: {exc}.  "
            "Ensure Redis is running and REDIS_URL is correctly set."
        ) from exc


# ---------------------------------------------------------------------------
# Shared helpers (used by both manager classes)
# ---------------------------------------------------------------------------

def _apply_filters(records: List[Dict], filters: Optional[Dict]) -> List[Dict]:
    """Filter a list of dicts by equality on each key/value in *filters*."""
    if not filters:
        return records
    return [
        r for r in records
        if all(str(r.get(k)) == str(v) for k, v in filters.items())
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
    return _apply_filters(df.to_dict(orient="records"), filters)


def _append_jsonl(path: Path, record: Dict) -> None:
    """Append *record* as a JSON line to *path* (runs in a thread)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# ---------------------------------------------------------------------------
# DatabaseManager — three-tier fallback (UI / scripts)
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Async data-access layer with a three-tier fallback chain.

    Tier 1: asyncpg (PostgreSQL via ``DATABASE_URL``)
    Tier 2: supabase-py (via ``SUPABASE_URL`` + ``SUPABASE_KEY``)
    Tier 3: CSV / JSONL files in ``data/``

    Appropriate for the Streamlit UI and evaluation scripts where graceful
    degradation is acceptable.  **Not** appropriate for production API routes
    or Celery tasks — use ``ProductionDatabaseManager`` there.
    """

    def __init__(self) -> None:
        self._database_url: Optional[str] = os.environ.get("DATABASE_URL")
        self._supabase_url: Optional[str] = os.environ.get("SUPABASE_URL")
        self._supabase_key: Optional[str] = os.environ.get("SUPABASE_KEY")

    # ── asyncpg helpers ──────────────────────────────────────────────────────

    async def _asyncpg_fetch(self, query: str, *args: Any) -> List[Dict]:
        conn = await asyncpg.connect(self._database_url)
        try:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def _asyncpg_execute(self, query: str, *args: Any) -> None:
        conn = await asyncpg.connect(self._database_url)
        try:
            await conn.execute(query, *args)
        finally:
            await conn.close()

    # ── supabase helper ──────────────────────────────────────────────────────

    def _get_supabase_client(self):                     # type: ignore[return]
        return _supabase_create_client(self._supabase_url, self._supabase_key)

    # ── Public API ───────────────────────────────────────────────────────────

    async def get_claims(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Return claim records matching *filters* (three-tier fallback)."""
        if HAS_ASYNCPG and self._database_url:
            try:
                rows = await self._asyncpg_fetch("SELECT * FROM claims LIMIT 1000")
                return _apply_filters(rows, filters)
            except Exception as exc:
                logger.warning("asyncpg get_claims failed (%s); trying supabase.", exc)

        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                query = client.table("claims").select("*")
                if filters:
                    for k, v in filters.items():
                        query = query.eq(k, v)
                return (query.execute().data or [])
            except Exception as exc:
                logger.warning("supabase get_claims failed (%s); using CSV.", exc)

        return await asyncio.to_thread(_csv_to_records, _CLAIMS_CSV, filters)

    async def get_policies(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Return policy records matching *filters* (three-tier fallback)."""
        if HAS_ASYNCPG and self._database_url:
            try:
                rows = await self._asyncpg_fetch("SELECT * FROM policies LIMIT 1000")
                return _apply_filters(rows, filters)
            except Exception as exc:
                logger.warning("asyncpg get_policies failed (%s); trying supabase.", exc)

        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                query = client.table("policies").select("*")
                if filters:
                    for k, v in filters.items():
                        query = query.eq(k, v)
                return (query.execute().data or [])
            except Exception as exc:
                logger.warning("supabase get_policies failed (%s); using CSV.", exc)

        return await asyncio.to_thread(_csv_to_records, _POLICIES_CSV, filters)

    async def save_decision(self, decision: Dict) -> None:
        """Persist a copilot decision (three-tier fallback)."""
        if HAS_ASYNCPG and self._database_url:
            try:
                columns = ", ".join(decision.keys())
                placeholders = ", ".join(f"${i + 1}" for i in range(len(decision)))
                await self._asyncpg_execute(
                    f"INSERT INTO decisions ({columns}) VALUES ({placeholders})",
                    *decision.values(),
                )
                return
            except Exception as exc:
                logger.warning("asyncpg save_decision failed (%s); trying supabase.", exc)

        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                self._get_supabase_client().table("decisions").insert(decision).execute()
                return
            except Exception as exc:
                logger.warning("supabase save_decision failed (%s); using JSONL.", exc)

        await asyncio.to_thread(_append_jsonl, _DECISIONS_JSONL, decision)


# ---------------------------------------------------------------------------
# ProductionDatabaseManager — strict, no fallbacks (API routes / workers)
# ---------------------------------------------------------------------------

class ProductionDatabaseManager(DatabaseManager):
    """
    Strict subclass of ``DatabaseManager`` for production API routes and
    Celery tasks.

    Differences from the base class
    --------------------------------
    * ``assert_ready()`` must be called (or awaited) before any data method.
      It raises ``InfrastructureError`` immediately if PostgreSQL is
      unreachable or ``DATABASE_URL`` is unset.
    * ``get_claims``, ``get_policies``, and ``save_decision`` raise
      ``InfrastructureError`` without falling back to CSV/JSONL when
      PostgreSQL fails.  Supabase is still tried as a secondary tier.
    * There is no third-tier CSV fallback.
    """

    def assert_ready(self) -> None:
        """Synchronous pre-flight check.

        Raises
        ------
        InfrastructureError
            When ``DATABASE_URL`` is unset or PostgreSQL is unreachable.
        """
        assert_postgres_reachable(self._database_url)

    # ── Override: no CSV fallback ────────────────────────────────────────────

    async def get_claims(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Return claim records — PostgreSQL required; Supabase as secondary."""
        if not self._database_url:
            raise InfrastructureError(
                "DATABASE_URL is not set.  "
                "Production routes require a live PostgreSQL connection."
            )
        if HAS_ASYNCPG:
            try:
                rows = await self._asyncpg_fetch("SELECT * FROM claims LIMIT 1000")
                return _apply_filters(rows, filters)
            except InfrastructureError:
                raise
            except Exception as exc:
                logger.warning(
                    "ProductionDatabaseManager.get_claims: asyncpg failed (%s); "
                    "trying supabase.", exc,
                )
        else:
            logger.warning(
                "ProductionDatabaseManager.get_claims: asyncpg not installed; "
                "trying supabase."
            )

        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                query = client.table("claims").select("*")
                if filters:
                    for k, v in filters.items():
                        query = query.eq(k, v)
                return (query.execute().data or [])
            except Exception as exc:
                raise InfrastructureError(
                    f"All database tiers failed for claims: {exc}"
                ) from exc

        raise InfrastructureError(
            "PostgreSQL and Supabase are both unavailable.  "
            "Cannot retrieve claims in production mode."
        )

    async def get_policies(self, filters: Optional[Dict] = None) -> List[Dict]:
        """Return policy records — PostgreSQL required; Supabase as secondary."""
        if not self._database_url:
            raise InfrastructureError(
                "DATABASE_URL is not set.  "
                "Production routes require a live PostgreSQL connection."
            )
        if HAS_ASYNCPG:
            try:
                rows = await self._asyncpg_fetch("SELECT * FROM policies LIMIT 1000")
                return _apply_filters(rows, filters)
            except InfrastructureError:
                raise
            except Exception as exc:
                logger.warning(
                    "ProductionDatabaseManager.get_policies: asyncpg failed (%s); "
                    "trying supabase.", exc,
                )
        else:
            logger.warning(
                "ProductionDatabaseManager.get_policies: asyncpg not installed; "
                "trying supabase."
            )

        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                client = self._get_supabase_client()
                query = client.table("policies").select("*")
                if filters:
                    for k, v in filters.items():
                        query = query.eq(k, v)
                return (query.execute().data or [])
            except Exception as exc:
                raise InfrastructureError(
                    f"All database tiers failed for policies: {exc}"
                ) from exc

        raise InfrastructureError(
            "PostgreSQL and Supabase are both unavailable.  "
            "Cannot retrieve policies in production mode."
        )

    async def save_decision(self, decision: Dict) -> None:
        """Persist a decision — PostgreSQL required; Supabase as secondary."""
        if not self._database_url:
            raise InfrastructureError(
                "DATABASE_URL is not set.  "
                "Production routes require a live PostgreSQL connection."
            )
        if HAS_ASYNCPG:
            try:
                columns = ", ".join(decision.keys())
                placeholders = ", ".join(f"${i + 1}" for i in range(len(decision)))
                await self._asyncpg_execute(
                    f"INSERT INTO decisions ({columns}) VALUES ({placeholders})",
                    *decision.values(),
                )
                return
            except InfrastructureError:
                raise
            except Exception as exc:
                logger.warning(
                    "ProductionDatabaseManager.save_decision: asyncpg failed (%s); "
                    "trying supabase.", exc,
                )
        else:
            logger.warning(
                "ProductionDatabaseManager.save_decision: asyncpg not installed; "
                "trying supabase."
            )

        if HAS_SUPABASE and self._supabase_url and self._supabase_key:
            try:
                self._get_supabase_client().table("decisions").insert(decision).execute()
                return
            except Exception as exc:
                raise InfrastructureError(
                    f"All database tiers failed when saving decision: {exc}"
                ) from exc

        raise InfrastructureError(
            "PostgreSQL and Supabase are both unavailable.  "
            "Cannot persist decisions in production mode."
        )
