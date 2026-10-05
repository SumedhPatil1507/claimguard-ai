"""
Role-Based Access Control (RBAC) for ClaimGuard AI.

API keys are resolved from the API_KEYS environment variable (a JSON
object mapping key → role) or fall back to a hard-coded demo dictionary.

FastAPI usage
-------------
    from src.rbac import require_role

    @router.get("/underwrite")
    async def underwrite(role=Depends(require_role(["admin", "analyst"]))):
        ...

Role tiers
----------
  admin   — full access (read + write + admin endpoints)
  analyst — read + claims/underwriting workflow
  viewer  — read-only, public dashboards
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, Optional

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------
ROLES: List[str] = ["admin", "analyst", "viewer"]

# ---------------------------------------------------------------------------
# API key → role mapping
# ---------------------------------------------------------------------------
_DEMO_KEYS: dict[str, str] = {
    "demo-admin": "admin",
    "demo-analyst": "analyst",
    "demo-viewer": "viewer",
}


def _load_api_keys() -> dict[str, str]:
    """Load API keys from the API_KEYS environment variable or use demo keys."""
    raw = os.environ.get("API_KEYS")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
            logger.warning("API_KEYS env var is not a JSON object; using demo keys.")
        except json.JSONDecodeError:
            logger.warning("API_KEYS env var is not valid JSON; using demo keys.")
    return _DEMO_KEYS.copy()


# Resolved once at module load; re-reads env each call via function.
# Kept as a module-level dict so it can be patched in tests if needed.
API_KEYS: dict[str, str] = _load_api_keys()


def get_role(api_key: str) -> Optional[str]:
    """Return the role for *api_key*, or ``None`` if the key is unknown."""
    return API_KEYS.get(api_key)


# ---------------------------------------------------------------------------
# FastAPI dependency factory
# ---------------------------------------------------------------------------


def require_role(allowed_roles: List[str]) -> Callable:
    """Return a FastAPI dependency that enforces role membership.

    Parameters
    ----------
    allowed_roles:
        List of role strings (subset of ``ROLES``) permitted to call the
        endpoint.

    Raises
    ------
    HTTPException 401 — key absent or not found in the API key registry.
    HTTPException 403 — key found but role not in *allowed_roles*.
    """

    async def _dependency(x_api_key: Optional[str] = Header(default=None)) -> str:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="X-API-Key header is required.")
        role = get_role(x_api_key)
        if role is None:
            raise HTTPException(status_code=401, detail="Invalid API key.")
        if role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{role}' is not permitted. Required: {allowed_roles}.",
            )
        return role

    return _dependency


# ---------------------------------------------------------------------------
# Optional Starlette middleware (logging only — primary enforcement via dep)
# ---------------------------------------------------------------------------
try:
    from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore
    from starlette.requests import Request as StarletteRequest  # type: ignore
    from starlette.responses import Response as StarletteResponse  # type: ignore

    class RBACMiddleware(BaseHTTPMiddleware):
        """Log the resolved RBAC role on every incoming request.

        This is a non-blocking observer; enforcement happens in the
        ``require_role`` dependency attached to individual routes.
        """

        async def dispatch(
            self, request: StarletteRequest, call_next: Callable
        ) -> StarletteResponse:
            api_key = request.headers.get("x-api-key") or request.headers.get(
                "X-API-Key"
            )
            role = get_role(api_key) if api_key else None
            logger.debug(
                "RBAC: path=%s method=%s role=%s",
                request.url.path,
                request.method,
                role or "anonymous",
            )
            response: StarletteResponse = await call_next(request)
            return response

except ImportError:
    # Starlette not available — define a no-op sentinel so imports don't fail.
    class RBACMiddleware:  # type: ignore[no-redef]
        """No-op placeholder used when Starlette is not installed."""

        pass
