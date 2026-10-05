"""
ClaimGuard AI — FastAPI application entry-point.

Endpoints
---------
GET  /health                  – liveness probe (no auth)
POST /underwrite              – underwriting risk scoring  (analyst|admin)
POST /claims/score            – claims fraud scoring       (analyst|admin)
POST /copilot/decide          – Policy Copilot pipeline    (analyst|admin)
GET  /graph/collusion-rings   – collusion ring detection   (analyst|admin)
GET  /compliance              – IRDAI compliance report    (viewer|analyst|admin)
GET  /metrics                 – Prometheus metrics         (admin)
"""

from __future__ import annotations

import json as _json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# JSON logging
# ---------------------------------------------------------------------------


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        return _json.dumps(
            {
                "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            }
        )


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler])
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core FastAPI imports (always available — listed in requirements.txt)
# ---------------------------------------------------------------------------
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import pandas as pd

# ---------------------------------------------------------------------------
# src imports — each wrapped so the API starts even with missing packages
# ---------------------------------------------------------------------------

try:
    from src.underwriting import UnderwritingEngine, UnderwritingFeatures, UnderwritingResult

    _uw_engine = UnderwritingEngine()
except Exception as _e:
    logger.warning(f"underwriting unavailable: {_e}")
    _uw_engine = None
    UnderwritingFeatures = None  # type: ignore[assignment,misc]
    UnderwritingResult = None  # type: ignore[assignment,misc]

try:
    from src.claims_fraud import FraudDetectionEngine, ClaimFeatures, FraudScoringResult

    _fraud_engine = FraudDetectionEngine()
except Exception as _e:
    logger.warning(f"claims_fraud unavailable: {_e}")
    _fraud_engine = None
    ClaimFeatures = None  # type: ignore[assignment,misc]
    FraudScoringResult = None  # type: ignore[assignment,misc]

try:
    from src.agent_graph import run_copilot
except Exception as _e:
    logger.warning(f"agent_graph unavailable: {_e}")
    run_copilot = None  # type: ignore[assignment]

try:
    from src.graph_collusion import GraphCollusionDetector
except Exception as _e:
    logger.warning(f"graph_collusion unavailable: {_e}")
    GraphCollusionDetector = None  # type: ignore[assignment,misc]

try:
    from src.compliance_irdai import generate_compliance_report
except Exception as _e:
    logger.warning(f"compliance_irdai unavailable: {_e}")
    generate_compliance_report = None  # type: ignore[assignment]

# Optional: src.rbac / src.rate_limit (graceful — we define our own inline)
try:
    from src.rbac import require_role as _src_require_role
except Exception:
    _src_require_role = None  # type: ignore[assignment]

try:
    from src.rate_limit import rate_limit_dependency
except Exception:
    rate_limit_dependency = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Demo API key → role mapping
# ---------------------------------------------------------------------------

_DEMO_KEYS: dict[str, str] = {
    "admin-key-demo": "admin",
    "analyst-key-demo": "analyst",
    "viewer-key-demo": "viewer",
}

# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def get_api_key(
    x_api_key: str = Header(None, alias="X-API-Key"),
) -> str:
    """Resolve the X-API-Key header to a role string or raise 401/403."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="X-API-Key header required")

    role = _DEMO_KEYS.get(x_api_key)

    if not role:
        # Also accept keys injected via the API_KEYS environment variable
        # (a JSON object mapping key → role).
        try:
            env_keys: dict = _json.loads(os.environ.get("API_KEYS", "{}"))
            role = env_keys.get(x_api_key)
        except Exception:
            pass

    if not role:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return role


def require_roles(allowed: list[str]):
    """Return a FastAPI dependency that enforces a role allowlist."""

    async def _check(role: str = Depends(get_api_key)) -> str:
        if role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"Role {role!r} is not permitted for this endpoint",
            )
        return role

    return _check


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ClaimGuard AI",
    version="1.0.0",
    description=(
        "Agentic InsurTech Platform for Underwriting Risk Scoring "
        "and Claims Fraud Detection"
    ),
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
    duration_ms = round((time.monotonic() - t0) * 1000, 2)
    logger.info(
        _json.dumps(
            {
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )
    )
    return response


# ---------------------------------------------------------------------------
# Fallback Pydantic models (used when src schemas could not be imported)
# ---------------------------------------------------------------------------


class _FallbackUnderwritingFeatures(BaseModel):
    age: int = 35
    annual_income: float = 500000.0
    credit_score: int = 700
    sum_insured: float = 1000000.0
    coverage_type: str = "comprehensive"
    num_dependents: int = 2
    prior_claims_count: int = 0
    region: str = "urban"
    occupation: str = "salaried"


class _FallbackClaimFeatures(BaseModel):
    claim_id: str = "CLM-000"
    claim_amount: float = 50000.0
    days_since_policy_start: int = 180
    num_prior_claims: int = 0
    claim_type: str = "motor"
    claim_severity: str = "medium"
    repair_shop_id: str | None = None
    medical_provider_id: str | None = None
    claimant_id: str = "CLT-000"


# Resolve which schema classes to advertise in the OpenAPI docs.
_UwFeaturesModel = UnderwritingFeatures if UnderwritingFeatures is not None else _FallbackUnderwritingFeatures
_ClaimFeaturesModel = ClaimFeatures if ClaimFeatures is not None else _FallbackClaimFeatures


# ---------------------------------------------------------------------------
# Copilot request schema
# ---------------------------------------------------------------------------


class CopilotRequest(BaseModel):
    query: str
    context_type: str = "underwriting"
    features: dict = {}
    session_id: str = ""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
async def health():
    """Liveness probe — no authentication required."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/underwrite", tags=["Underwriting"])
async def underwrite(
    features: _UwFeaturesModel,  # type: ignore[valid-type]
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """Score an applicant at policy-issuance time and return a risk tier."""
    if _uw_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Underwriting engine is unavailable (missing dependencies or training data).",
        )
    try:
        result = _uw_engine.predict(features)
        # Pydantic v2 — use model_dump(); fall back to dict() for older objects.
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)
    except Exception as exc:
        logger.warning(f"underwrite: prediction failed — {exc}")
        raise HTTPException(status_code=500, detail=f"Underwriting prediction error: {exc}")


@app.post("/claims/score", tags=["Claims"])
async def claims_score(
    features: _ClaimFeaturesModel,  # type: ignore[valid-type]
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """Score a claim at filing time and return a fraud probability."""
    if _fraud_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection engine is unavailable (missing dependencies or training data).",
        )
    try:
        result = _fraud_engine.predict(features)
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)
    except Exception as exc:
        logger.warning(f"claims/score: prediction failed — {exc}")
        raise HTTPException(status_code=500, detail=f"Fraud scoring error: {exc}")


@app.post("/copilot/decide", tags=["Policy Copilot"])
async def copilot_decide(
    body: CopilotRequest,
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """
    Run the Policy Copilot LangGraph pipeline and return the full state dict.

    The pipeline always enqueues the draft for human analyst review (HITL).
    It never auto-approves or auto-denies a claim or policy.
    """
    if run_copilot is None:
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
        logger.warning(f"copilot/decide: pipeline failed — {exc}")
        raise HTTPException(status_code=500, detail=f"Policy Copilot error: {exc}")


@app.get("/graph/collusion-rings", tags=["Graph Intelligence"])
async def graph_collusion_rings(
    role: str = Depends(require_roles(["analyst", "admin"])),
):
    """Detect claim-ring collusion from the sample claims dataset."""
    if GraphCollusionDetector is None:
        raise HTTPException(
            status_code=503,
            detail="Graph collusion detector is unavailable.",
        )

    csv_path = Path(__file__).parent.parent / "data" / "sample_claims.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not load claims data from {csv_path}: {exc}",
        )

    try:
        rings = GraphCollusionDetector().analyze(df)
        return {"rings": [r.model_dump() for r in rings]}
    except Exception as exc:
        logger.warning(f"graph/collusion-rings: analysis failed — {exc}")
        raise HTTPException(status_code=500, detail=f"Graph analysis error: {exc}")


@app.get("/compliance", tags=["Compliance"])
async def compliance(
    role: str = Depends(require_roles(["viewer", "analyst", "admin"])),
):
    """Return the IRDAI compliance report with score and per-control detail."""
    if generate_compliance_report is None:
        raise HTTPException(
            status_code=503,
            detail="Compliance reporting module is unavailable.",
        )
    try:
        report = generate_compliance_report()
        return report.model_dump() if hasattr(report, "model_dump") else dict(report)
    except Exception as exc:
        logger.warning(f"compliance: report generation failed — {exc}")
        raise HTTPException(status_code=500, detail=f"Compliance report error: {exc}")


@app.get("/metrics", tags=["Observability"])
async def metrics(
    role: str = Depends(require_roles(["admin"])),
):
    """Expose Prometheus metrics in text format."""
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return PlainTextResponse(
            generate_latest().decode(),
            media_type=CONTENT_TYPE_LATEST,
        )
    except ImportError:
        return JSONResponse({"message": "prometheus_client not installed"}, status_code=200)


# ---------------------------------------------------------------------------
# Startup event — warm up ML engines in a background thread
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    logger.info("ClaimGuard AI API starting up")

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


# ---------------------------------------------------------------------------
# Main block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
