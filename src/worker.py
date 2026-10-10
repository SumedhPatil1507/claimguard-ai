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
  frontend/client layer.

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

import json as _json
import logging
import os
from typing import Any, Dict, List, Optional

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
# Task helpers & Redis Pub/Sub Event Dispatcher
# ---------------------------------------------------------------------------

def get_task_channel(task_id: str) -> str:
    """Return the Redis Pub/Sub channel name for real-time task SSE streaming."""
    return f"claimguard:task_events:{task_id}"


def publish_task_event(task_id: str, event: Dict[str, Any]) -> None:
    """Publish a real-time task lifecycle transition to Redis Pub/Sub."""
    if not task_id:
        return
    channel = get_task_channel(task_id)
    try:
        import redis as _r
        r = _r.from_url(REDIS_URL, socket_timeout=1.0)
        r.publish(channel, _json.dumps(event))
    except Exception as exc:
        logger.debug("Failed to publish task event to Redis Pub/Sub: %s", exc)


def _safe_update_state(task_self: Any, state: str, meta: Dict[str, Any]) -> None:
    """Update Celery task state safely without breaking in test/eager execution modes."""
    try:
        if task_self and hasattr(task_self, "update_state"):
            task_self.update_state(state=state, meta=meta)
    except Exception as exc:
        logger.debug("update_state ignored (e.g. offline backend/test mode): %s", exc)


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
# Standardized ExplainableDecision attachment (v1.9 output contract)
# ---------------------------------------------------------------------------

def _inline_similarity_scores(features: Optional[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """
    Optional Qdrant hybrid retrieval for scoring responses.

    Opt-in via ``CLAIMGUARD_INLINE_RETRIEVAL=1`` (enabled in docker-compose,
    disabled by default so tests / offline workers never load embedding
    models).  Returns ``[{source, score, chunk_id}, ...]`` or ``None``.
    """
    if not features or os.environ.get("CLAIMGUARD_INLINE_RETRIEVAL", "0") != "1":
        return None
    try:
        from src.vector_store import policy_store
        query = " ".join(
            str(v) for v in features.values() if isinstance(v, (str, int, float))
        )[:300]
        if not query.strip():
            return None
        hits = policy_store.search(query, top_k=3)
        return [
            {
                "source": h.get("source", "policy_doc"),
                "score": h.get("score", 0.0),
                "chunk_id": h.get("chunk_id") or (
                    (h.get("payload") or {}).get("chunk_id")
                    if isinstance(h.get("payload"), dict)
                    else None
                ),
            }
            for h in hits
        ]
    except Exception as exc:
        logger.debug("inline retrieval unavailable: %s", exc)
        return None


def _attach_decision(
    kind: str,
    res_dict: Dict[str, Any],
    features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach the standardized ``decision`` ExplainableDecision JSON
    (verdict / reasoning / recommendation / next_steps) to a scoring result.
    Failures are logged and never break the scoring pipeline.
    """
    try:
        from src.decision_json import build_fraud_decision, build_underwriting_decision
        similarity = _inline_similarity_scores(features)
        if kind == "underwriting":
            res_dict["decision"] = build_underwriting_decision(
                res_dict, similarity
            ).model_dump()
        else:
            claim_type = str((features or {}).get("claim_type", "motor"))
            res_dict["decision"] = build_fraud_decision(
                res_dict, similarity, claim_type=claim_type
            ).model_dump()
    except Exception as exc:
        logger.warning("_attach_decision failed (%s): %s", kind, exc)
    return res_dict



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
        """
        task_id = self.request.id or ""

        # ── 1. Infrastructure checks ────────────────────────────────────────
        try:
            _assert_redis_reachable()
            _assert_postgres_reachable()
        except RuntimeError as exc:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": {"stage": "Infrastructure error", "percent": 100},
                "error": str(exc),
                "error_type": "RuntimeError",
            })
            raise     # hard failure — no retry

        _safe_update_state(
            self,
            state="STARTED",
            meta={"stage": "Validating infrastructure & applicant features", "percent": 25},
        )
        publish_task_event(task_id, {
            "task_id": task_id,
            "status": "STARTED",
            "progress": {"stage": "Validating infrastructure & applicant features", "percent": 25},
        })

        # ── 2. Import and run ML engine ─────────────────────────────────────
        try:
            _safe_update_state(
                self,
                state="PROGRESS",
                meta={"stage": "Evaluating underwriting risk features & SHAP attribution", "percent": 70},
            )
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "PROGRESS",
                "progress": {"stage": "Evaluating underwriting risk features & SHAP attribution", "percent": 70},
            })

            from src.underwriting import UnderwritingEngine, UnderwritingFeatures

            engine = UnderwritingEngine()
            parsed = UnderwritingFeatures(**features)
            result = engine.predict(parsed)
            res_dict = _serialise(result)
            _attach_decision("underwriting", res_dict, features)

            logger.info(
                "score_underwriting_task: task_id=%s risk_tier=%s",
                task_id,
                getattr(result, "risk_tier", "unknown"),
            )

            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "SUCCESS",
                "progress": {"stage": "Completed", "percent": 100},
                "result": res_dict,
            })
            return res_dict

        except SoftTimeLimitExceeded as exc:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": {"stage": "Task timed out", "percent": 100},
                "error": "Task execution timed out",
                "error_type": "SoftTimeLimitExceeded",
            })
            raise     # propagate timeout as failure

        except Exception as exc:
            logger.warning(
                "score_underwriting_task: transient error — %s (attempt %d/%d)",
                exc,
                self.request.retries + 1,
                _MAX_RETRIES + 1,
            )
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "RETRY" if self.request.retries < _MAX_RETRIES else "FAILURE",
                "progress": {"stage": f"Error: {exc}", "percent": 100},
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            })
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
        """
        task_id = self.request.id or ""

        # ── 1. Infrastructure checks ────────────────────────────────────────
        try:
            _assert_redis_reachable()
            _assert_postgres_reachable()
        except RuntimeError as exc:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": {"stage": "Infrastructure error", "percent": 100},
                "error": str(exc),
                "error_type": "RuntimeError",
            })
            raise     # hard failure — no retry

        _safe_update_state(
            self,
            state="STARTED",
            meta={"stage": "Validating infrastructure & claims payload", "percent": 25},
        )
        publish_task_event(task_id, {
            "task_id": task_id,
            "status": "STARTED",
            "progress": {"stage": "Validating infrastructure & claims payload", "percent": 25},
        })

        # ── 2. Import and run ML engine ─────────────────────────────────────
        try:
            _safe_update_state(
                self,
                state="PROGRESS",
                meta={"stage": "Scoring fraud probability & computing feature drivers", "percent": 70},
            )
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "PROGRESS",
                "progress": {"stage": "Scoring fraud probability & computing feature drivers", "percent": 70},
            })

            from src.claims_fraud import ClaimFeatures, FraudDetectionEngine

            engine = FraudDetectionEngine()
            parsed = ClaimFeatures(**features)
            result = engine.predict(parsed)
            res_dict = _serialise(result)
            _attach_decision("claims", res_dict, features)

            logger.info(
                "score_claim_task: task_id=%s fraud_flag=%s fraud_score=%.3f",
                task_id,
                getattr(result, "fraud_flag", "unknown"),
                float(getattr(result, "fraud_score", 0)),
            )

            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "SUCCESS",
                "progress": {"stage": "Completed", "percent": 100},
                "result": res_dict,
            })
            return res_dict

        except SoftTimeLimitExceeded as exc:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": {"stage": "Task timed out", "percent": 100},
                "error": "Task execution timed out",
                "error_type": "SoftTimeLimitExceeded",
            })
            raise

        except Exception as exc:
            logger.warning(
                "score_claim_task: transient error — %s (attempt %d/%d)",
                exc,
                self.request.retries + 1,
                _MAX_RETRIES + 1,
            )
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "RETRY" if self.request.retries < _MAX_RETRIES else "FAILURE",
                "progress": {"stage": f"Error: {exc}", "percent": 100},
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            })
            raise self.retry(exc=exc, countdown=_RETRY_BACKOFF * (2 ** self.request.retries))

    # -----------------------------------------------------------------------
    # CSV batch ingestion task (POST /ingest/file → Celery)
    # -----------------------------------------------------------------------

    @celery_app.task(
        name="claimguard.ingest_csv_batch",
        bind=True,
        max_retries=_MAX_RETRIES,
        default_retry_delay=_RETRY_BACKOFF,
    )
    def ingest_csv_batch(
        self,
        *,
        rows: List[Dict[str, Any]],
        kind: str = "generic",
        filename: str = "upload.csv",
    ) -> Dict[str, Any]:
        """
        Background task: batch-process an uploaded CSV.

        * ``kind='claims'``   — score every row with the fraud engine.
        * ``kind='policies'`` — score every row with the underwriting engine.
        * ``kind='generic'``  — validate/archive only (no scoring engine).

        Publishes STARTED / PROGRESS / SUCCESS / FAILURE events to Redis
        Pub/Sub so ``GET /tasks/stream/{task_id}`` can stream live progress.

        Returns
        -------
        dict — ``{filename, kind, rows_received, processed, flagged, errors,
        items, decision}`` where ``decision`` is the standardized
        ExplainableDecision JSON summary of the batch.
        """
        task_id = self.request.id or ""

        # ── 1. Infrastructure checks ────────────────────────────────────────
        try:
            _assert_redis_reachable()
        except RuntimeError as exc:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": {"stage": "Infrastructure error", "percent": 100},
                "error": str(exc),
                "error_type": "RuntimeError",
            })
            raise     # hard failure — no retry

        total = len(rows or [])
        publish_task_event(task_id, {
            "task_id": task_id,
            "status": "STARTED",
            "progress": {"stage": f"Parsing {total} rows from {filename}", "percent": 10},
        })
        _safe_update_state(
            self,
            state="STARTED",
            meta={"stage": f"Parsing {total} rows from {filename}", "percent": 10},
        )
        # ── 2. Score rows with the appropriate engine ───────────────────────
        items: List[Dict[str, Any]] = []
        errors = 0
        flagged = 0
        processed = 0
        step = max(1, total // 10)

        try:
            engine = None
            features_cls = None
            if kind == "claims":
                from src.claims_fraud import ClaimFeatures, FraudDetectionEngine
                engine, features_cls = FraudDetectionEngine(), ClaimFeatures
            elif kind == "policies":
                from src.underwriting import UnderwritingEngine, UnderwritingFeatures
                engine, features_cls = UnderwritingEngine(), UnderwritingFeatures

            for idx, row in enumerate(rows or []):
                try:
                    if engine is not None and features_cls is not None:
                        parsed = features_cls(**row)
                        result = engine.predict(parsed)
                        res_dict = _serialise(result)
                        res_dict = _attach_decision(kind, res_dict, row)
                        decision = res_dict.get("decision") or {}
                        is_flagged = (
                            bool(res_dict.get("fraud_flag"))
                            or str(res_dict.get("risk_tier", "")).lower() == "high"
                            or str(decision.get("recommendation", "")) != "Auto-Approve"
                        )
                        items.append({
                            "row": idx,
                            "score": res_dict,
                            "id": res_dict.get("claim_id") or idx,
                        })
                        processed += 1
                        if is_flagged:
                            flagged += 1
                    else:
                        items.append({"row": idx, "score": None})
                        processed += 1
                except Exception as row_exc:      # per-row validation failure
                    errors += 1
                    items.append({"row": idx, "error": str(row_exc)})

                if (idx + 1) % step == 0 or (idx + 1) == total:
                    pct = 10 + int(80 * (idx + 1) / max(total, 1))
                    publish_task_event(task_id, {
                        "task_id": task_id,
                        "status": "PROGRESS",
                        "progress": {
                            "stage": f"Scoring row {idx + 1}/{total} "
                                     f"({flagged} flagged, {errors} errors)",
                            "percent": pct,
                        },
                    })
                    _safe_update_state(
                        self,
                        state="PROGRESS",
                        meta={
                            "stage": f"Scoring row {idx + 1}/{total}",
                            "percent": pct,
                        },
                    )

            # ── 3. Standardized batch decision summary ──────────────────────
            from src.decision_json import build_batch_decision
            summary: Dict[str, Any] = {
                "filename": filename,
                "kind": kind,
                "rows_received": total,
                "processed": processed,
                "flagged": flagged,
                "errors": errors,
                "items": items[:1000],
            }
            try:
                summary["decision"] = build_batch_decision(
                    kind, processed, flagged, errors
                ).model_dump()
            except Exception as exc:
                logger.warning("ingest_csv_batch: decision build failed: %s", exc)

            logger.info(
                "ingest_csv_batch: task_id=%s kind=%s processed=%d flagged=%d errors=%d",
                task_id, kind, processed, flagged, errors,
            )
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "SUCCESS",
                "progress": {"stage": "Completed", "percent": 100},
                "result": summary,
            })
            return summary

        except SoftTimeLimitExceeded:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE",
                "progress": {"stage": "Task timed out", "percent": 100},
                "error": "Batch ingestion timed out",
                "error_type": "SoftTimeLimitExceeded",
            })
            raise
        except Exception as exc:
            publish_task_event(task_id, {
                "task_id": task_id,
                "status": "FAILURE" if self.request.retries >= _MAX_RETRIES else "RETRY",
                "progress": {"stage": f"Error: {exc}", "percent": 100},
                "error": str(exc),
                "error_type": exc.__class__.__name__,
            })
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

    def ingest_csv_batch(**kwargs: Any) -> None:  # type: ignore[misc]
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
    PROGRESS  — task is in progress with intermediate execution details
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
        payload["progress"] = {"stage": "Completed", "percent": 100}
    elif state == "FAILURE":
        exc = ar.result
        payload["error"] = str(exc) if exc else "Unknown error"
        # Scrub tracebacks that may contain sensitive path information
        payload["error_type"] = type(exc).__name__ if exc else "Unknown"
        payload["progress"] = {"stage": "Failed", "percent": 100}
    elif state in ("STARTED", "PROGRESS", "RETRY"):
        info = ar.info or {}
        if isinstance(info, dict):
            payload["progress"] = info

    return payload
