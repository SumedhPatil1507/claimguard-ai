"""
ClaimGuard AI — FastAPI application
====================================

Endpoint summary
----------------
GET  /health                     Liveness probe (no auth)
POST /underwrite                 Enqueue underwriting job → {task_id, status_url}
POST /claims/score               Enqueue fraud-scoring job → {task_id, status_url}
GET  /tasks/{task_id}            Poll Celery task status
POST /copilot/decide             Run Policy Copilot pipeline inline
GET  /graph/collusion-rings      Detect collusion rings (PostgreSQL required)
GET  /compliance                 IRDAI compliance report
GET  /metrics                    Prometheus metrics (admin only)

Async execution model
---------------------
/underwrite and /claims/score are **non-blocking**: they validate the request
body, check Redis reachability, enqueue a Celery task, and immediately return
HTTP 202 with a ``task_id`` and a ``status_url`` the caller can poll.

/tasks/{task_id} returns the current state:
  PENDING  — queued, worker has not started yet
  STARTED  — worker is running the inference
  SUCCESS  — complete; ``result`` key contains the payload
  FAILURE  — failed; ``error`` and ``error_type`` keys explain why
  RETRY    — worker retrying after a transient error

Infrastructure errors
---------------------
When Redis or PostgreSQL is unreachable, affected endpoints return
503 Service Unavailable immediately — there is no silent CSV fallback in
production routes.
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before anything else so env vars are available to all imports
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# JSON structured logging
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        return _json.dumps({
            "time":    datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        })

_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI core (always available)
# ---------------------------------------------------------------------------
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager

# ---------------------------------------------------------------------------
# Infrastructure: Celery worker + database checks
# ---------------------------------------------------------------------------
try:
    from src.worker import (
        HAS_CELERY,
        celery_app,
        get_task_result,
        score_claim_task,
        score_underwriting_task,
    )
except Exception as _e:
    logger.warning("worker import failed: %s", _e)
    HAS_CELERY = False
    celery_app = None
    get_task_result = None          # type: ignore[assignment]
    score_underwriting_task = None  # type: ignore[assignment]
    score_claim_task = None         # type: ignore[assignment]

try:
    from src.database import (
        InfrastructureError,
        assert_postgres_reachable,
        assert_redis_reachable,
    )
except Exception as _e:
    logger.warning("database import failed: %s", _e)

    class InfrastructureError(RuntimeError):  # type: ignore[no-redef]
        pass

    def assert_postgres_reachable(*_a, **_kw) -> None:   # type: ignore[misc]
        pass

    def assert_redis_reachable(*_a, **_kw) -> None:      # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Domain module imports — graceful on missing deps
# ---------------------------------------------------------------------------

try:
    from src.underwriting import UnderwritingFeatures, UnderwritingResult
    _HAS_UW_SCHEMA = True
except Exception as _e:
    logger.warning("underwriting schema unavailable: %s", _e)
    _HAS_UW_SCHEMA = False
    UnderwritingFeatures = None  # type: ignore[assignment,misc]

try:
    from src.claims_fraud import ClaimFeatures, FraudScoringResult
    _HAS_CLAIM_SCHEMA = True
except Exception as _e:
    logger.warning("claims_fraud schema unavailable: %s", _e)
    _HAS_CLAIM_SCHEMA = False
    ClaimFeatures = None         # type: ignore[assignment,misc]

try:
    from src.agent_graph import run_copilot
    _HAS_COPILOT = True
except Exception as _e:
    logger.warning("agent_graph unavailable: %s", _e)
    _HAS_COPILOT = False
    run_copilot = None           # type: ignore[assignment]

try:
    from src.graph_collusion import GraphCollusionDetector
    _HAS_GRAPH = True
except Exception as _e:
    logger.warning("graph_collusion unavailable: %s", _e)
    _HAS_GRAPH = False
    GraphCollusionDetector = None  # type: ignore[assignment,misc]

try:
    from src.compliance_irdai import generate_compliance_report
    _HAS_COMPLIANCE = True
except Exception as _e:
    logger.warning("compliance_irdai unavailable: %s", _e)
    _HAS_COMPLIANCE = False
    generate_compliance_report = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Demo API key → role mapping
# (also picks up keys from the API_KEYS env-var JSON object)
# ---------------------------------------------------------------------------
_DEMO_KEYS: Dict[str, str] = {
    "admin-key-demo":   "admin",
    "analyst-key-demo": "analyst",
    "viewer-key-demo":  "viewer",
}


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> str:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required.")
    role = _DEMO_KEYS.get(x_api_key)
    if not role:
        try:
            env_keys: dict = _json.loads(os.environ.get("API_KEYS", "{}"))
            role = env_keys.get(x_api_key)
        except Exception:
            pass
    if not role:
        raise HTTPException(status_code=403, detail="Invalid API key.")
    return role


def require_roles(allowed: list[str]):
    """Dependency factory: enforce role membership."""
    async def _check(role: str = Depends(get_api_key)) -> str:
        if role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Role {role!r} is not permitted. Required: {allowed}.",
            )
        return role
    return _check


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------

class _FallbackUnderwritingFeatures(BaseModel):
    age: int = 35
    annual_income: float = 500_000.0
    credit_score: int = 700
    sum_insured: float = 1_000_000.0
    coverage_type: str = "motor"
    num_dependents: int = 2
    prior_claims_count: int = 0
    region: str = "north"
    occupation: str = "salaried"


class _FallbackClaimFeatures(BaseModel):
    claim_id: str = "CLM-000"
    claimant_id: str = "CLT-000"
    policy_id: str = "POL-000"
    claim_amount: float = 50_000.0
    days_since_policy_start: int = 180
    num_prior_claims: int = 0
    claim_type: str = "motor"
    claim_severity: str = "medium"
    repair_shop_id: str | None = None
    medical_provider_id: str | None = None


class TaskEnqueueResponse(BaseModel):
    """Returned by /underwrite and /claims/score (HTTP 202)."""
    task_id: str = Field(..., description="UUID of the enqueued Celery task.")
    status:  str = Field("PENDING", description="Initial task state.")
    status_url: str = Field(..., description="URL to poll for task result.")
    message: str = Field(..., description="Human-readable confirmation.")


class TaskStatusResponse(BaseModel):
    """Returned by GET /tasks/{task_id}."""
    task_id:    str
    status:     str
    result:     Any = None
    error:      str | None = None
    error_type: str | None = None
    progress:   dict | None = None


class CopilotRequest(BaseModel):
    query:        str
    context_type: str  = "underwriting"
    features:     dict = {}
    session_id:   str  = ""


# Resolve which schema classes to advertise in OpenAPI
_UwFeaturesModel    = UnderwritingFeatures  if _HAS_UW_SCHEMA    else _FallbackUnderwritingFeatures
_ClaimFeaturesModel = ClaimFeatures         if _HAS_CLAIM_SCHEMA  else _FallbackClaimFeatures


# ---------------------------------------------------------------------------
# Lifespan: warm up ML engines in a background thread
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Warm-up on startup; clean up on shutdown."""
    logger.info("ClaimGuard AI API starting — version 1.0.0")
    import threading

    def _warmup() -> None:
        try:
            from src.underwriting import UnderwritingEngine
            UnderwritingEngine()
        except Exception:
            pass
        try:
            from src.claims_fraud import FraudDetectionEngine
            FraudDetectionEngine()
        except Exception:
            pass

    threading.Thread(target=_warmup, daemon=True).start()
    yield
    logger.info("ClaimGuard AI API shutting down")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ClaimGuard AI",
    version="1.0.0",
    description=(
        "Agentic InsurTech Platform — async inference via Celery + Redis. "
        "POST to /underwrite or /claims/score to enqueue; "
        "GET /tasks/{task_id} to poll."
    ),
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request-logging middleware
# ---------------------------------------------------------------------------

@app.middleware("http")
async def _log_requests(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    logger.info(_json.dumps({
        "method":      request.method,
        "path":        request.url.path,
        "status_code": response.status_code,
        "duration_ms": round((time.monotonic() - t0) * 1000, 2),
    }))
    return response


# ---------------------------------------------------------------------------
# Helper: build a status URL
# ---------------------------------------------------------------------------

def _status_url(request: Request, task_id: str) -> str:
    """Build an absolute URL to GET /tasks/{task_id}."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/tasks/{task_id}"


# ---------------------------------------------------------------------------
# Helper: map InfrastructureError → 503
# ---------------------------------------------------------------------------

def _infra_503(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=str(exc))


# ===========================================================================
# Endpoints
# ===========================================================================

# ── /health ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health():
    """Liveness probe — no authentication required."""
    return {
        "status":    "ok",
        "version":   "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── POST /underwrite (async, enqueue) ───────────────────────────────────────

@app.post(
    "/underwrite",
    status_code=202,
    response_model=TaskEnqueueResponse,
    tags=["Underwriting"],
    summary="Enqueue an underwriting risk-scoring job",
)
async def underwrite(
    features: _UwFeaturesModel,      # type: ignore[valid-type]
    request: Request,
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """
    Validate the underwriting feature payload, check Redis reachability, and
    enqueue a Celery task.

    Returns HTTP **202 Accepted** immediately with a ``task_id``.
    Poll ``status_url`` (``GET /tasks/{task_id}``) until ``status`` is
    ``SUCCESS`` or ``FAILURE``.

    Raises **503** if Redis is unreachable.
    Raises **503** if the Celery worker package is not installed.
    """
    if not HAS_CELERY or score_underwriting_task is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Celery worker is not available.  "
                "Install it with: pip install 'celery[redis]' "
                "and ensure a Redis broker is running."
            ),
        )

    # Hard Redis check — fail fast before touching the queue
    try:
        assert_redis_reachable()
    except InfrastructureError as exc:
        raise _infra_503(exc)

    task_id = str(uuid.uuid4())
    features_dict = (
        features.model_dump() if hasattr(features, "model_dump") else dict(features)
    )

    try:
        score_underwriting_task.apply_async(
            kwargs={"features": features_dict},
            task_id=task_id,
        )
    except Exception as exc:
        logger.error("underwrite: failed to enqueue task — %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue underwriting task: {exc}",
        )

    logger.info("underwrite: enqueued task_id=%s role=%s", task_id, role)
    return TaskEnqueueResponse(
        task_id=task_id,
        status="PENDING",
        status_url=_status_url(request, task_id),
        message=(
            f"Underwriting job enqueued.  "
            f"Poll {_status_url(request, task_id)} for the result."
        ),
    )


# ── POST /claims/score (async, enqueue) ─────────────────────────────────────

@app.post(
    "/claims/score",
    status_code=202,
    response_model=TaskEnqueueResponse,
    tags=["Claims"],
    summary="Enqueue a claims fraud-scoring job",
)
async def claims_score(
    features: _ClaimFeaturesModel,   # type: ignore[valid-type]
    request: Request,
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """
    Validate the claim feature payload, check Redis reachability, and enqueue
    a Celery task.

    Returns HTTP **202 Accepted** immediately with a ``task_id``.
    Poll ``status_url`` (``GET /tasks/{task_id}``) until ``status`` is
    ``SUCCESS`` or ``FAILURE``.

    Raises **503** if Redis is unreachable.
    Raises **503** if the Celery worker package is not installed.
    """
    if not HAS_CELERY or score_claim_task is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Celery worker is not available.  "
                "Install it with: pip install 'celery[redis]' "
                "and ensure a Redis broker is running."
            ),
        )

    try:
        assert_redis_reachable()
    except InfrastructureError as exc:
        raise _infra_503(exc)

    task_id = str(uuid.uuid4())
    features_dict = (
        features.model_dump() if hasattr(features, "model_dump") else dict(features)
    )

    try:
        score_claim_task.apply_async(
            kwargs={"features": features_dict},
            task_id=task_id,
        )
    except Exception as exc:
        logger.error("claims/score: failed to enqueue task — %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to enqueue claims scoring task: {exc}",
        )

    logger.info("claims/score: enqueued task_id=%s role=%s", task_id, role)
    return TaskEnqueueResponse(
        task_id=task_id,
        status="PENDING",
        status_url=_status_url(request, task_id),
        message=(
            f"Claims scoring job enqueued.  "
            f"Poll {_status_url(request, task_id)} for the result."
        ),
    )


# ── GET /tasks/{task_id} (poll) ─────────────────────────────────────────────

@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    tags=["Tasks"],
    summary="Poll the status of an enqueued task",
)
async def task_status(
    task_id: str,
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """
    Return the current state of the Celery task identified by *task_id*.

    Possible ``status`` values
    --------------------------
    * ``PENDING``  — queued; worker has not started yet
    * ``STARTED``  — worker is running the inference
    * ``SUCCESS``  — complete; ``result`` contains the payload
    * ``FAILURE``  — failed; ``error`` and ``error_type`` explain why
    * ``RETRY``    — worker is retrying after a transient error

    Raises **503** if Celery is not installed.
    Raises **503** if Redis is unreachable.
    Raises **404** if *task_id* is unknown (PENDING in Celery terms — the
    caller can distinguish by checking whether they themselves enqueued it).
    """
    if not HAS_CELERY or get_task_result is None:
        raise HTTPException(
            status_code=503,
            detail="Celery is not installed; task status polling is unavailable.",
        )

    try:
        payload = get_task_result(task_id)
    except InfrastructureError as exc:
        raise _infra_503(exc)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.error("tasks/%s: status retrieval failed — %s", task_id, exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve task status: {exc}",
        )

    return TaskStatusResponse(**payload)


# ── POST /copilot/decide ─────────────────────────────────────────────────────

@app.post("/copilot/decide", tags=["Policy Copilot"])
async def copilot_decide(
    body: CopilotRequest,
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """
    Run the Policy Copilot LangGraph pipeline **inline** (synchronous).

    The pipeline always enqueues its draft for human analyst review (HITL).
    It never auto-approves or auto-denies.

    Note: for very large feature sets this call may be slow.  A future
    version will support async enqueue like the ML endpoints.
    """
    if not _HAS_COPILOT or run_copilot is None:
        raise HTTPException(
            status_code=503,
            detail="Policy Copilot is unavailable (agent_graph import failed).",
        )
    try:
        state = run_copilot(
            query=body.query,
            context_type=body.context_type,
            features=body.features,
            session_id=body.session_id or None,
        )
        return state
    except Exception as exc:
        logger.warning("copilot/decide: pipeline error — %s", exc)
        raise HTTPException(status_code=500, detail=f"Policy Copilot error: {exc}")


# ── GET /graph/collusion-rings ───────────────────────────────────────────────

@app.get("/graph/collusion-rings", tags=["Graph Intelligence"])
async def graph_collusion_rings(
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """
    Detect claim-ring collusion from PostgreSQL claims data.

    Raises **503** if the GraphCollusionDetector is unavailable.
    Raises **503** if PostgreSQL is unreachable (no CSV fallback in production).
    """
    if not _HAS_GRAPH or GraphCollusionDetector is None:
        raise HTTPException(
            status_code=503,
            detail="Graph collusion detector is unavailable.",
        )

    # Hard database check — no CSV fallback in the API layer
    try:
        assert_postgres_reachable()
    except InfrastructureError as exc:
        raise _infra_503(exc)

    # Load claims from PostgreSQL
    try:
        from src.database import ProductionDatabaseManager
        import pandas as pd
        db = ProductionDatabaseManager()
        records = await db.get_claims()
        if not records:
            raise HTTPException(
                status_code=404,
                detail="No claims records found in the database.",
            )
        df = pd.DataFrame(records)
    except InfrastructureError as exc:
        raise _infra_503(exc)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("graph/collusion-rings: data load failed — %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Could not load claims data: {exc}",
        )

    try:
        rings = GraphCollusionDetector().analyze(df)
        return {"rings": [r.model_dump() for r in rings]}
    except Exception as exc:
        logger.warning("graph/collusion-rings: analysis failed — %s", exc)
        raise HTTPException(status_code=500, detail=f"Graph analysis error: {exc}")


# ── GET /compliance ──────────────────────────────────────────────────────────

@app.get("/compliance", tags=["Compliance"])
async def compliance(
    role: str = Depends(require_roles(["viewer", "analyst", "admin"])),
):
    """Return the IRDAI compliance report with score and per-control detail."""
    if not _HAS_COMPLIANCE or generate_compliance_report is None:
        raise HTTPException(
            status_code=503,
            detail="Compliance reporting module is unavailable.",
        )
    try:
        report = generate_compliance_report()
        return report.model_dump() if hasattr(report, "model_dump") else dict(report)
    except Exception as exc:
        logger.warning("compliance: report error — %s", exc)
        raise HTTPException(status_code=500, detail=f"Compliance report error: {exc}")


# ── GET /metrics ─────────────────────────────────────────────────────────────

@app.get("/metrics", tags=["Observability"])
async def metrics(
    role: str = Depends(require_roles(["admin"])),
):
    """Expose Prometheus metrics in text format (admin only)."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)
    except ImportError:
        return JSONResponse({"message": "prometheus_client not installed"})


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
