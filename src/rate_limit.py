"""
Sliding-window rate limiter for ClaimGuard AI.

Uses ``collections.deque`` and ``time.monotonic`` for an O(1)-amortised
per-key sliding window with no external dependencies.

FastAPI usage
-------------
    from src.rate_limit import rate_limit_dependency

    @router.get("/copilot/decide")
    async def decide(_: None = Depends(rate_limit_dependency())):
        ...
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable, Dict

from fastapi import HTTPException, Request


class RateLimiter:
    """Sliding-window rate limiter keyed by an arbitrary string.

    Parameters
    ----------
    max_requests:
        Maximum number of requests allowed within *window_seconds*.
    window_seconds:
        Width of the sliding time window in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._buckets: Dict[str, deque] = {}

    def check_rate_limit(self, key: str) -> bool:
        """Check and record a request for *key*.

        Returns
        -------
        bool
            ``True`` when the request is within the allowed rate;
            ``False`` when the key has exceeded the limit and the caller
            should return HTTP 429.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        if key not in self._buckets:
            self._buckets[key] = deque()

        bucket = self._buckets[key]

        # Evict timestamps that have fallen outside the window.
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self.max_requests:
            return False

        bucket.append(now)
        return True


# ---------------------------------------------------------------------------
# Module-level default limiter instance (shared across all routes)
# ---------------------------------------------------------------------------
_default_limiter = RateLimiter(max_requests=100, window_seconds=60)


def rate_limit_dependency(
    limiter: RateLimiter | None = None,
) -> Callable:
    """Return a FastAPI dependency that enforces rate limiting by client IP.

    Parameters
    ----------
    limiter:
        A :class:`RateLimiter` instance to use.  Defaults to the module-level
        shared limiter (100 req / 60 s).

    Raises
    ------
    HTTPException 429 — client has exceeded the configured rate.
    """
    _limiter = limiter or _default_limiter

    async def _dependency(request: Request) -> None:
        client_ip: str = (
            request.client.host if request.client else "unknown"
        )
        if not _limiter.check_rate_limit(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry after a short wait.",
            )

    return _dependency
