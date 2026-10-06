"""
src/worker.py
=============
Celery application and background task definitions for ClaimGuard AI.

Design decisions
----------------
* **Hard infrastructure checks** — tasks fail immediately with a clear error
  when Redis is unreachable (broker ping) or when PostgreSQL is unreachable
  (DATABASE_URL is set but the connection cannot be established).  There are
  no silent CSV/in-memory fallbacks inside tasks; those belong only in the
  Streamlit UI layer.

* **Idempotency** — each task accepts a caller-supplied ``task_id`` (UUID)
  that is stored as the Celery task id via ``apply_async(task_id=...)``.
  Callers can therefore poll the same id without races.

* **Result serialisation** — all return values are plain ``dict`` objects
  (JSON-serialisable) so Celery's JSON result backend works without pickle.

* **Retry policy** — transient ML errors (e.g. model loading race) are
  retried up to ``MAX_RETRIES`` times with exponential back-off.  Hard infra
  errors (Redis/Postgres unreachable) are *not* retried; they surface as
  FAILURE immediately.

* **Worker startup** — the module is importable without a live Redis; the
  Celery app is configured lazily.  Import-time failures surface as a clear
  ``RuntimeError`` only when ``celery_app`` is first accessed.

Environment variables
---------------------
REDIS_URL           Redis DSN used for both broker and result backend.
                    Default: redis://localhost:6379/0
DATABASE_URL        asyncpg-compatible PostgreSQL DSN (optional).  When set,
                    tasks verify the connection is live before running.
CELERY_TASK_TIMEOUT Hard time limit per task in seconds.  Default: 300.

Usage
-----
    # Start worker
    celery -A src.worker worker --loglevel=info --concurrency=4

    # Enqueue from application code
    from src.worker import score_underwriting_task, score_claim_task
    job = score_underwriting_task.apply_async(
        kwargs={"features": {...}},
        task_id="<uuid>",
    )
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis / Celery availability guard
# ---------------------------------------------------------------------------
try:
    from celery import Celery                           # type: ignore
    from celery.exceptions import SoftTimeLimitExceeded  # type: ignore
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False
    Celery = None                                       # type: ignore[misc,assignment]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
_TASK_TIMEOUT: int = int(os.environ.get("CELERY_TASK_TIMEOUT", "300"))
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: int = 5      # seconds; doubles each retry


# ---------------------------------------------------------------------------
# Celery application factory
# ---------------------------------------------------------------------------

def _make_celery_app() -> "Celery":
    """Create and configure the Celery application.

    Raises
    ------
    RuntimeError
        When celery or redis packages are not installed.
    """
    if not HAS_CELERY:
        raise RuntimeError(
            "celery package is not installed.  "
            "Install it with:  pip install 'celery[redis]'"
        )

    app = Celery(
        "claimguard",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )
    app.conf.update(
        # Serialisation
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # Reliability
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Timeouts
        task_soft_time_limit=_TASK_TIMEOUT,
        task_time_limit=_TASK_TIMEOUT + 30,
        # Result expiry — keep results for 24 h so polling always works
        result_expires=86_400,
        # Visibility timeout for Redis broker (2× hard timeout)
        broker_transport_options={
            "visibility_timeout": (_TASK_TIMEOUT + 30) * 2,
        },
        # Worker behaviour
        worker_prefetch_multiplier=1,
        task_track_started=True,
    )
    return app


# Module-level Celery app — lazily initialised on first attribute access
# so that importing this module never raises even when Redis is absent.
celery_app: "Celery" = _make_celery_app() if HAS_CELERY else None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Infrastructure health checks (called synchronously inside each task)
# ---------------------------------------------------------------------------

def _assert_redis_reachable() -> None:
    """Ping the Redis broker.

    Raises
    ------
    RuntimeError
        When the broker is unreachable.  The task should surface this as
        FAILURE without retry.
    """
    try:
        import redis as _redis     # type: ignore
        client = _redis.from_url(REDIS_URL, socket_connect_timeout=3)
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Redis broker is unreachable at {REDIS_URL!r}: {exc}.  "
            "Ensure Redis is running and REDIS_URL is correctly set."
        ) from exc


def _assert_postgres_reachable() -> None:
    """Verify the PostgreSQL connection if DATABASE_URL is configured.

    This runs a synchronous connection attempt using the asyncpg-compatible
    DSN via psycopg2 (if available) or a raw socket check as a fallback.

    Raises
    ------
    RuntimeError
        When DATABASE_URL is set but the server cannot be reached.
    """
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return   # no Postgres configured — nothing to check

    # Try psycopg2 first (fast, zero-overhead)
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(db_url, connect_timeout=3)
        conn.close()
        return
    except ImportError:
        pass
    except Exception as exc:
        raise RuntimeError(
            f"PostgreSQL is unreachable (psycopg2): {exc}.  "
            "Ensure DATABASE_URL is valid and the server is running."
        ) from exc

    # Fallback: raw socket check using the host:port from the DSN
    try:
        import socket
        from urllib.parse import urlparse
        parsed = urlparse(db_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 5432
        with socket.create_connection((host, port), timeout=3):
            pass
    except Exception as exc:
        raise RuntimeError(
            f"PostgreSQL is unreachable ({host}:{port}): {exc}.  "
            "Ensure DATABASE_URL is valid and the server is running."
        ) from exc


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def _serialise(obj: Any) -> Any:
    """Convert a Pydantic v2/v1 model or arbitrary object to a JSON-safe dict."""
    if hasattr(obj, "model_dump"):      # Pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):            # Pydantic v1
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return str(obj)                     # last resort


# ---------------------------------------------------------------------------
# Celery tasks
# ---------------------------------------------------------------------------

if HAS_CELERY and celery_app is not None:

    @celery_app.task(
        name="claimguard.score_underwriting",
        bind=True,
        max_retries=_MAX_RETRIES,
        default_retry_delay=_RETRY_BACKOFF,
    )
    def score_underwriting_task(self, *, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Background task: run underwriting risk scoring.

        Parameters
        ----------
        features : dict
            JSON-serialised ``UnderwritingFeatures`` field values.

        Returns
        -------
        dict
            JSON-serialised ``UnderwritingResult``.

        Raises
        ------
        RuntimeError
            On hard infrastructure failures (Redis/Postgres unreachable).
            These are NOT retried.
        celery.exceptions.Retry
            On transient ML errors — retried up to MAX_RETRIES times.
        """
        # ── 1. Infrastructure checks ────────────────────────────────────────
        try:
            _assert_redis_reachable()
            _assert_postgres_reachable()
        except RuntimeError:
            raise     # hard failure — no retry

        # ── 2. Import and run ML engine ─────────────────────────────────────
        try:
            from src.underwriting import UnderwritingEngine, UnderwritingFeatures

            engine = UnderwritingEngine()
            parsed = UnderwritingFeatures(**features)
            result = engine.predict(parsed)
            logger.info(
                "score_underwriting_task: task_id=%s risk_tier=%s",
                self.request.id,
                getattr(result, "risk_tier", "unknown"),
            )
            return _serialise(result)

        except SoftTimeLimitExceeded:
            raise     # propagate timeout as failure

        except Exception as exc:
            logger.warning(
                "score_underwriting_task: transient error — %s (attempt %d/%d)",
                exc,
                self.request.retries + 1,
                _MAX_RETRIES + 1,
            )
            raise self.retry(exc=exc, countdown=_RETRY_BACKOFF * (2 ** self.request.retries))

    @celery_app.task(
        name="claimguard.score_claim",
        bind=True,
        max_retries=_MAX_RETRIES,
        default_retry_delay=_RETRY_BACKOFF,
    )
    def score_claim_task(self, *, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Background task: run claims fraud scoring.

        Parameters
        ----------
        features : dict
            JSON-serialised ``ClaimFeatures`` field values.

        Returns
        -------
        dict
            JSON-serialised ``FraudScoringResult``.

        Raises
        ------
        RuntimeError
            On hard infrastructure failures (Redis/Postgres unreachable).
            These are NOT retried.
        celery.exceptions.Retry
            On transient ML errors — retried up to MAX_RETRIES times.
        """
        # ── 1. Infrastructure checks ────────────────────────────────────────
        try:
            _assert_redis_reachable()
            _assert_postgres_reachable()
        except RuntimeError:
            raise     # hard failure — no retry

        # ── 2. Import and run ML engine ─────────────────────────────────────
        try:
            from src.claims_fraud import ClaimFeatures, FraudDetectionEngine

            engine = FraudDetectionEngine()
            parsed = ClaimFeatures(**features)
            result = engine.predict(parsed)
            logger.info(
                "score_claim_task: task_id=%s fraud_flag=%s fraud_score=%.3f",
                self.request.id,
                getattr(result, "fraud_flag", "unknown"),
                float(getattr(result, "fraud_score", 0)),
            )
            return _serialise(result)

        except SoftTimeLimitExceeded:
            raise

        except Exception as exc:
            logger.warning(
                "score_claim_task: transient error — %s (attempt %d/%d)",
                exc,
                self.request.retries + 1,
                _MAX_RETRIES + 1,
            )
            raise self.retry(exc=exc, countdown=_RETRY_BACKOFF * (2 ** self.request.retries))

else:
    # ── Stubs when Celery is not installed ───────────────────────────────────
    # These exist so that ``from src.worker import score_underwriting_task``
    # never raises ImportError.  Calling them raises RuntimeError immediately.

    def score_underwriting_task(**kwargs: Any) -> None:  # type: ignore[misc]
        raise RuntimeError(
            "celery package is not installed.  "
            "Install it with:  pip install 'celery[redis]'"
        )

    def score_claim_task(**kwargs: Any) -> None:  # type: ignore[misc]
        raise RuntimeError(
            "celery package is not installed.  "
            "Install it with:  pip install 'celery[redis]'"
        )


# ---------------------------------------------------------------------------
# Convenience: get AsyncResult without importing celery in callers
# ---------------------------------------------------------------------------

def get_task_result(task_id: str) -> Dict[str, Any]:
    """
    Return a status dict for *task_id* that is safe to serialise to JSON.

    States
    ------
    PENDING   — task queued, not yet started
    STARTED   — task is running on a worker
    SUCCESS   — task completed; ``result`` key contains the payload
    FAILURE   — task failed; ``error`` key contains the error message
    RETRY     — task is being retried after a transient error

    Raises
    ------
    RuntimeError
        When Celery is not installed or Redis is unreachable.
    """
    if not HAS_CELERY or celery_app is None:
        raise RuntimeError("celery package is not installed.")

    _assert_redis_reachable()

    from celery.result import AsyncResult   # type: ignore

    ar: AsyncResult = celery_app.AsyncResult(task_id)
    state: str = ar.state

    payload: Dict[str, Any] = {"task_id": task_id, "status": state}

    if state == "SUCCESS":
        payload["result"] = ar.result
    elif state == "FAILURE":
        exc = ar.result
        payload["error"] = str(exc) if exc else "Unknown error"
        # Scrub tracebacks that may contain sensitive path information
        payload["error_type"] = type(exc).__name__ if exc else "Unknown"
    elif state in ("STARTED", "RETRY"):
        info = ar.info or {}
        if isinstance(info, dict):
            payload["progress"] = info

    return payload
