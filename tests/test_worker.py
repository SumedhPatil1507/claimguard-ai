"""
tests/test_worker.py
====================
Unit tests for:

  src/worker.py          — Celery app, task functions, get_task_result()
  src/database.py        — InfrastructureError, assert_postgres_reachable(),
                           assert_redis_reachable(), ProductionDatabaseManager
  api/main.py            — POST /underwrite, POST /claims/score (202),
                           GET /tasks/{task_id}, 503 error paths

No live Redis or PostgreSQL required.  Every network call is replaced by a
unittest.mock patch.

Test groups
-----------
A  InfrastructureError & connectivity helpers (database.py)
B  ProductionDatabaseManager — no-fallback behaviour
C  worker.py — _assert_redis_reachable / _assert_postgres_reachable
D  worker.py — task execution (Celery task with mocked ML engines)
E  worker.py — get_task_result() status translation
F  api/main.py — POST /underwrite  (happy path, 202)
G  api/main.py — POST /claims/score (happy path, 202)
H  api/main.py — GET /tasks/{task_id}
I  api/main.py — 503 paths (Redis down, Celery absent)
J  api/main.py — /health always returns 200
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure repo root is on sys.path so all src.* imports resolve
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------
try:
    import celery  # noqa: F401
    HAS_CELERY = True
except ImportError:
    HAS_CELERY = False

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

_UW_FEATURES: Dict[str, Any] = {
    "age": 35,
    "annual_income": 800_000.0,
    "credit_score": 720,
    "sum_insured": 1_000_000.0,
    "coverage_type": "motor",
    "num_dependents": 2,
    "prior_claims_count": 0,
    "region": "north",
    "occupation": "salaried",
}

_CLAIM_FEATURES: Dict[str, Any] = {
    "claim_id": "CLM-TEST-001",
    "claimant_id": "CLT-TEST-001",
    "policy_id": "POL-TEST-001",
    "claim_amount": 75_000.0,
    "days_since_policy_start": 90,
    "num_prior_claims": 1,
    "claim_type": "motor",
    "claim_severity": "medium",
    "repair_shop_id": "SHOP-001",
    "medical_provider_id": None,
}

_UW_RESULT: Dict[str, Any] = {
    "risk_tier": "low",
    "risk_score": 0.22,
    "premium_adjustment": 0.95,
    "shap_drivers": [],
    "model_version": "xgb-lgbm-v1",
    "timestamp": "2024-01-01T00:00:00+00:00",
}

_FRAUD_RESULT: Dict[str, Any] = {
    "claim_id": "CLM-TEST-001",
    "fraud_score": 0.31,
    "fraud_flag": False,
    "confidence_tier": "medium",
    "shap_drivers": [],
    "model_version": "xgb-lgbm-v1",
    "timestamp": "2024-01-01T00:00:00+00:00",
}


# ===========================================================================
# A — InfrastructureError & connectivity helpers
# ===========================================================================

class TestInfrastructureError:
    def test_is_runtime_error_subclass(self) -> None:
        from src.database import InfrastructureError
        assert issubclass(InfrastructureError, RuntimeError)

    def test_can_be_raised_and_caught(self) -> None:
        from src.database import InfrastructureError
        with pytest.raises(InfrastructureError, match="test message"):
            raise InfrastructureError("test message")


class TestAssertRedisReachable:
    def test_raises_when_redis_package_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import database as db_mod
        monkeypatch.setattr(db_mod, "HAS_REDIS", False)
        from src.database import InfrastructureError, assert_redis_reachable
        with pytest.raises(InfrastructureError, match="redis package"):
            assert_redis_reachable("redis://localhost:6379/0")

    def test_raises_when_ping_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import database as db_mod
        monkeypatch.setattr(db_mod, "HAS_REDIS", True)
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("refused")
        mock_from_url = MagicMock(return_value=mock_client)
        monkeypatch.setattr(db_mod._redis_lib, "from_url", mock_from_url)
        from src.database import InfrastructureError, assert_redis_reachable
        with pytest.raises(InfrastructureError, match="Redis is unreachable"):
            assert_redis_reachable("redis://localhost:6379/0")

    def test_passes_when_ping_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src import database as db_mod
        monkeypatch.setattr(db_mod, "HAS_REDIS", True)
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        monkeypatch.setattr(db_mod._redis_lib, "from_url", MagicMock(return_value=mock_client))
        from src.database import assert_redis_reachable
        assert_redis_reachable("redis://localhost:6379/0")   # must not raise


class TestAssertPostgresReachable:
    def test_raises_when_url_not_set(self) -> None:
        from src.database import InfrastructureError, assert_postgres_reachable
        with pytest.raises(InfrastructureError, match="DATABASE_URL is not configured"):
            assert_postgres_reachable("")

    def test_raises_when_socket_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import socket as socket_mod
        monkeypatch.setattr(
            socket_mod, "create_connection",
            MagicMock(side_effect=OSError("connection refused")),
        )
        # Also make sure psycopg2 is not importable in this path
        import builtins
        real_import = builtins.__import__

        def _block_psycopg2(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_psycopg2)
        from src.database import InfrastructureError, assert_postgres_reachable
        with pytest.raises(InfrastructureError, match="TCP check failed"):
            assert_postgres_reachable("postgresql://user:pass@localhost:5432/db")

    def test_passes_when_socket_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import socket as socket_mod
        import builtins
        real_import = builtins.__import__

        def _block_psycopg2(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_psycopg2)
        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        monkeypatch.setattr(socket_mod, "create_connection", MagicMock(return_value=mock_sock))
        from src.database import assert_postgres_reachable
        assert_postgres_reachable("postgresql://user:pass@localhost:5432/db")  # no raise


# ===========================================================================
# B — ProductionDatabaseManager
# ===========================================================================

class TestProductionDatabaseManager:
    def test_assert_ready_raises_when_no_database_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.database import InfrastructureError, ProductionDatabaseManager
        mgr = ProductionDatabaseManager()
        with pytest.raises(InfrastructureError, match="DATABASE_URL"):
            mgr.assert_ready()

    def test_get_claims_raises_when_no_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.database import InfrastructureError, ProductionDatabaseManager
        mgr = ProductionDatabaseManager()
        with pytest.raises(InfrastructureError, match="DATABASE_URL is not set"):
            asyncio.run(mgr.get_claims())

    def test_get_policies_raises_when_no_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.database import InfrastructureError, ProductionDatabaseManager
        mgr = ProductionDatabaseManager()
        with pytest.raises(InfrastructureError, match="DATABASE_URL is not set"):
            asyncio.run(mgr.get_policies())

    def test_save_decision_raises_when_no_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        monkeypatch.delenv("DATABASE_URL", raising=False)
        from src.database import InfrastructureError, ProductionDatabaseManager
        mgr = ProductionDatabaseManager()
        with pytest.raises(InfrastructureError, match="DATABASE_URL is not set"):
            asyncio.run(mgr.save_decision({"key": "value"}))

    def test_get_claims_raises_on_asyncpg_failure_no_supabase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        monkeypatch.delenv("SUPABASE_URL", raising=False)
        from src import database as db_mod
        monkeypatch.setattr(db_mod, "HAS_ASYNCPG", True)
        monkeypatch.setattr(db_mod, "HAS_SUPABASE", False)
        from src.database import InfrastructureError, ProductionDatabaseManager
        mgr = ProductionDatabaseManager()
        with patch.object(mgr, "_asyncpg_fetch", side_effect=ConnectionError("pg down")):
            with pytest.raises(InfrastructureError, match="both unavailable"):
                asyncio.run(mgr.get_claims())

    def test_get_claims_no_csv_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ProductionDatabaseManager must never fall back to CSV."""
        import asyncio
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost/db")
        monkeypatch.setattr("src.database.HAS_ASYNCPG", True)
        monkeypatch.setattr("src.database.HAS_SUPABASE", False)
        from src.database import InfrastructureError, ProductionDatabaseManager
        mgr = ProductionDatabaseManager()
        with patch.object(mgr, "_asyncpg_fetch", side_effect=ConnectionError("down")):
            with pytest.raises(InfrastructureError):
                asyncio.run(mgr.get_claims())


# ===========================================================================
# C — worker._assert_redis_reachable / _assert_postgres_reachable
# ===========================================================================

@pytest.mark.skipif(not HAS_CELERY, reason="celery not installed")
class TestWorkerInfraChecks:
    def test_assert_redis_reachable_raises_on_connection_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.worker as wk
        mock_client = MagicMock()
        mock_client.ping.side_effect = ConnectionError("redis down")
        with patch("src.worker._redis_lib" if hasattr(wk, "_redis_lib") else
                   "redis.from_url", MagicMock(return_value=mock_client)):
            # Patch at the module level where worker imports redis
            with patch("redis.from_url", return_value=mock_client):
                with pytest.raises(RuntimeError, match="unreachable"):
                    wk._assert_redis_reachable()

    def test_assert_postgres_reachable_skips_when_no_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DATABASE_URL", raising=False)
        import src.worker as wk
        wk._assert_postgres_reachable()   # must not raise when URL absent

    def test_assert_postgres_reachable_raises_when_socket_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@badhost:5432/db")
        import socket as socket_mod
        import builtins
        real_import = builtins.__import__

        def _block_psycopg2(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_psycopg2)
        monkeypatch.setattr(
            socket_mod, "create_connection",
            MagicMock(side_effect=OSError("refused")),
        )
        import src.worker as wk
        with pytest.raises(RuntimeError, match="unreachable"):
            wk._assert_postgres_reachable()


# ===========================================================================
# D — Task execution (score_underwriting_task, score_claim_task)
# ===========================================================================

@pytest.mark.skipif(not HAS_CELERY, reason="celery not installed")
class TestCeleryTasks:
    """
    Tests run tasks **synchronously** using Celery's ALWAYS_EAGER setting so
    no broker or worker process is required.
    """

    @pytest.fixture(autouse=True)
    def _eager_mode(self) -> None:
        """Make Celery execute tasks inline without a broker."""
        from src.worker import celery_app
        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = True
        yield
        celery_app.conf.task_always_eager = False

    def _patch_infra(self):
        """Context manager: patch both infra checks to be no-ops."""
        import src.worker as wk
        return (
            patch.object(wk, "_assert_redis_reachable"),
            patch.object(wk, "_assert_postgres_reachable"),
        )

    def test_score_underwriting_task_returns_dict(self) -> None:
        from src.worker import score_underwriting_task

        mock_result = MagicMock()
        mock_result.model_dump = MagicMock(return_value=_UW_RESULT)

        mock_engine = MagicMock()
        mock_engine.predict.return_value = mock_result

        import src.worker as wk
        p_redis, p_pg = self._patch_infra()
        with p_redis, p_pg:
            with patch("src.worker.UnderwritingEngine", return_value=mock_engine, create=True):
                with patch("src.worker.UnderwritingFeatures", side_effect=lambda **kw: kw, create=True):
                    with patch("src.underwriting.UnderwritingEngine", return_value=mock_engine):
                        with patch("src.underwriting.UnderwritingFeatures") as MockFeatures:
                            MockFeatures.return_value = MagicMock()
                            # Run via apply (eager)
                            result = score_underwriting_task.apply(
                                kwargs={"features": _UW_FEATURES}
                            )
                            assert isinstance(result.result, dict)

    def test_score_claim_task_returns_dict(self) -> None:
        from src.worker import score_claim_task

        mock_result = MagicMock()
        mock_result.model_dump = MagicMock(return_value=_FRAUD_RESULT)

        mock_engine = MagicMock()
        mock_engine.predict.return_value = mock_result

        import src.worker as wk
        p_redis, p_pg = self._patch_infra()
        with p_redis, p_pg:
            with patch("src.claims_fraud.FraudDetectionEngine", return_value=mock_engine):
                with patch("src.claims_fraud.ClaimFeatures") as MockFeatures:
                    MockFeatures.return_value = MagicMock()
                    result = score_claim_task.apply(
                        kwargs={"features": _CLAIM_FEATURES}
                    )
                    assert isinstance(result.result, dict)

    def test_score_underwriting_hard_redis_failure_does_not_retry(self) -> None:
        """A RuntimeError from infra check must propagate without retry."""
        from src.worker import score_underwriting_task
        import src.worker as wk

        with patch.object(wk, "_assert_redis_reachable", side_effect=RuntimeError("redis down")):
            with pytest.raises(RuntimeError, match="redis down"):
                score_underwriting_task.apply(kwargs={"features": _UW_FEATURES})

    def test_score_claim_hard_postgres_failure_propagates(self) -> None:
        from src.worker import score_claim_task
        import src.worker as wk

        with patch.object(wk, "_assert_redis_reachable"):
            with patch.object(wk, "_assert_postgres_reachable", side_effect=RuntimeError("pg down")):
                with pytest.raises(RuntimeError, match="pg down"):
                    score_claim_task.apply(kwargs={"features": _CLAIM_FEATURES})


# ===========================================================================
# E — get_task_result() status translation
# ===========================================================================

@pytest.mark.skipif(not HAS_CELERY, reason="celery not installed")
class TestGetTaskResult:
    def _mock_async_result(self, state: str, result=None, info=None):
        ar = MagicMock()
        ar.state = state
        ar.result = result
        ar.info = info
        return ar

    def test_pending_state(self) -> None:
        from src.worker import celery_app, get_task_result
        ar = self._mock_async_result("PENDING")
        with patch.object(celery_app, "AsyncResult", return_value=ar):
            with patch("src.worker._assert_redis_reachable"):
                payload = get_task_result("test-id-001")
        assert payload["status"] == "PENDING"
        assert payload["task_id"] == "test-id-001"
        assert "result" not in payload

    def test_success_state_includes_result(self) -> None:
        from src.worker import celery_app, get_task_result
        ar = self._mock_async_result("SUCCESS", result=_UW_RESULT)
        with patch.object(celery_app, "AsyncResult", return_value=ar):
            with patch("src.worker._assert_redis_reachable"):
                payload = get_task_result("test-id-002")
        assert payload["status"] == "SUCCESS"
        assert payload["result"] == _UW_RESULT

    def test_failure_state_includes_error(self) -> None:
        from src.worker import celery_app, get_task_result
        exc = ValueError("model exploded")
        ar = self._mock_async_result("FAILURE", result=exc)
        with patch.object(celery_app, "AsyncResult", return_value=ar):
            with patch("src.worker._assert_redis_reachable"):
                payload = get_task_result("test-id-003")
        assert payload["status"] == "FAILURE"
        assert "model exploded" in payload["error"]
        assert payload["error_type"] == "ValueError"

    def test_started_state_includes_progress(self) -> None:
        from src.worker import celery_app, get_task_result
        ar = self._mock_async_result("STARTED", info={"step": "embedding"})
        with patch.object(celery_app, "AsyncResult", return_value=ar):
            with patch("src.worker._assert_redis_reachable"):
                payload = get_task_result("test-id-004")
        assert payload["status"] == "STARTED"
        assert payload.get("progress") == {"step": "embedding"}

    def test_raises_when_redis_unreachable(self) -> None:
        from src.worker import get_task_result
        with patch("src.worker._assert_redis_reachable", side_effect=RuntimeError("redis gone")):
            with pytest.raises(RuntimeError, match="redis gone"):
                get_task_result("any-id")

    def test_raises_when_celery_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.worker as wk
        monkeypatch.setattr(wk, "HAS_CELERY", False)
        monkeypatch.setattr(wk, "celery_app", None)
        with pytest.raises(RuntimeError, match="celery package"):
            wk.get_task_result("any-id")


# ===========================================================================
# Fixtures for FastAPI TestClient
# ===========================================================================

@pytest.fixture(scope="module")
def api_client():
    """TestClient with Redis and Celery fully mocked out."""
    if not HAS_FASTAPI:
        pytest.skip("fastapi not installed")

    # Patch assert_redis_reachable to be a no-op everywhere
    with patch("api.main.assert_redis_reachable"), \
         patch("api.main.assert_postgres_reachable"):
        from api.main import app
        from fastapi.testclient import TestClient
        yield TestClient(app, raise_server_exceptions=False)


_ANALYST_HEADERS = {"X-API-Key": "analyst-key-demo"}
_ADMIN_HEADERS   = {"X-API-Key": "admin-key-demo"}
_VIEWER_HEADERS  = {"X-API-Key": "viewer-key-demo"}


# ===========================================================================
# F — POST /underwrite
# ===========================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestUnderwriteEndpoint:
    def test_returns_202_with_task_id(self, api_client) -> None:
        mock_task = MagicMock()
        mock_task.id = "uw-task-uuid-001"
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable"), \
             patch("api.main.score_underwriting_task") as mock_enqueue:
            mock_enqueue.apply_async.return_value = mock_task
            resp = api_client.post("/underwrite", json=_UW_FEATURES, headers=_ANALYST_HEADERS)

        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "PENDING"
        assert "/tasks/" in body["status_url"]
        assert "message" in body

    def test_task_id_is_uuid_format(self, api_client) -> None:
        import re
        mock_task = MagicMock()
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable"), \
             patch("api.main.score_underwriting_task") as mock_enqueue:
            mock_enqueue.apply_async.return_value = mock_task
            resp = api_client.post("/underwrite", json=_UW_FEATURES, headers=_ADMIN_HEADERS)

        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        assert uuid_re.match(resp.json()["task_id"])

    def test_returns_401_without_api_key(self, api_client) -> None:
        resp = api_client.post("/underwrite", json=_UW_FEATURES)
        assert resp.status_code == 401

    def test_returns_403_for_viewer_role(self, api_client) -> None:
        resp = api_client.post("/underwrite", json=_UW_FEATURES, headers=_VIEWER_HEADERS)
        assert resp.status_code == 403

    def test_returns_503_when_celery_absent(self, api_client) -> None:
        with patch("api.main.HAS_CELERY", False), \
             patch("api.main.score_underwriting_task", None):
            resp = api_client.post(
                "/underwrite", json=_UW_FEATURES, headers=_ANALYST_HEADERS
            )
        assert resp.status_code == 503

    def test_returns_503_when_redis_unreachable(self, api_client) -> None:
        from src.database import InfrastructureError
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable",
                   side_effect=InfrastructureError("redis down")):
            resp = api_client.post(
                "/underwrite", json=_UW_FEATURES, headers=_ANALYST_HEADERS
            )
        assert resp.status_code == 503
        assert "redis" in resp.json()["detail"].lower()

    def test_returns_422_for_invalid_payload(self, api_client) -> None:
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable"):
            resp = api_client.post(
                "/underwrite",
                json={"age": "not-a-number"},
                headers=_ANALYST_HEADERS,
            )
        assert resp.status_code == 422

    def test_status_url_points_to_tasks_endpoint(self, api_client) -> None:
        mock_task = MagicMock()
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable"), \
             patch("api.main.score_underwriting_task") as mock_enqueue:
            mock_enqueue.apply_async.return_value = mock_task
            resp = api_client.post("/underwrite", json=_UW_FEATURES, headers=_ANALYST_HEADERS)

        task_id = resp.json()["task_id"]
        assert resp.json()["status_url"].endswith(f"/tasks/{task_id}")


# ===========================================================================
# G — POST /claims/score
# ===========================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestClaimsScoreEndpoint:
    def test_returns_202_with_task_id(self, api_client) -> None:
        mock_task = MagicMock()
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable"), \
             patch("api.main.score_claim_task") as mock_enqueue:
            mock_enqueue.apply_async.return_value = mock_task
            resp = api_client.post(
                "/claims/score", json=_CLAIM_FEATURES, headers=_ANALYST_HEADERS
            )

        assert resp.status_code == 202
        body = resp.json()
        assert "task_id" in body
        assert body["status"] == "PENDING"
        assert "/tasks/" in body["status_url"]

    def test_returns_503_when_redis_unreachable(self, api_client) -> None:
        from src.database import InfrastructureError
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable",
                   side_effect=InfrastructureError("redis gone")):
            resp = api_client.post(
                "/claims/score", json=_CLAIM_FEATURES, headers=_ANALYST_HEADERS
            )
        assert resp.status_code == 503

    def test_returns_503_when_celery_absent(self, api_client) -> None:
        with patch("api.main.HAS_CELERY", False), \
             patch("api.main.score_claim_task", None):
            resp = api_client.post(
                "/claims/score", json=_CLAIM_FEATURES, headers=_ANALYST_HEADERS
            )
        assert resp.status_code == 503

    def test_returns_401_without_api_key(self, api_client) -> None:
        resp = api_client.post("/claims/score", json=_CLAIM_FEATURES)
        assert resp.status_code == 401

    def test_returns_403_for_viewer(self, api_client) -> None:
        resp = api_client.post(
            "/claims/score", json=_CLAIM_FEATURES, headers=_VIEWER_HEADERS
        )
        assert resp.status_code == 403

    def test_different_task_id_each_call(self, api_client) -> None:
        """Each enqueue call must generate a fresh UUID."""
        mock_task = MagicMock()
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.assert_redis_reachable"), \
             patch("api.main.score_claim_task") as mock_enqueue:
            mock_enqueue.apply_async.return_value = mock_task
            r1 = api_client.post(
                "/claims/score", json=_CLAIM_FEATURES, headers=_ANALYST_HEADERS
            )
            r2 = api_client.post(
                "/claims/score", json=_CLAIM_FEATURES, headers=_ANALYST_HEADERS
            )
        assert r1.json()["task_id"] != r2.json()["task_id"]


# ===========================================================================
# H — GET /tasks/{task_id}
# ===========================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestTaskStatusEndpoint:
    def test_returns_pending_status(self, api_client) -> None:
        pending_payload = {"task_id": "abc-123", "status": "PENDING"}
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.get_task_result", return_value=pending_payload):
            resp = api_client.get("/tasks/abc-123", headers=_ANALYST_HEADERS)
        assert resp.status_code == 200
        assert resp.json()["status"] == "PENDING"

    def test_returns_success_with_result(self, api_client) -> None:
        success_payload = {
            "task_id": "abc-456",
            "status": "SUCCESS",
            "result": _UW_RESULT,
        }
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.get_task_result", return_value=success_payload):
            resp = api_client.get("/tasks/abc-456", headers=_ANALYST_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "SUCCESS"
        assert body["result"]["risk_tier"] == "low"

    def test_returns_failure_with_error(self, api_client) -> None:
        failure_payload = {
            "task_id": "abc-789",
            "status": "FAILURE",
            "error": "Model weights corrupt",
            "error_type": "ValueError",
        }
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.get_task_result", return_value=failure_payload):
            resp = api_client.get("/tasks/abc-789", headers=_ANALYST_HEADERS)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "FAILURE"
        assert body["error"] == "Model weights corrupt"
        assert body["error_type"] == "ValueError"

    def test_returns_503_when_redis_unreachable(self, api_client) -> None:
        from src.database import InfrastructureError
        with patch("api.main.HAS_CELERY", True), \
             patch("api.main.get_task_result",
                   side_effect=InfrastructureError("redis down")):
            resp = api_client.get("/tasks/any-id", headers=_ANALYST_HEADERS)
        assert resp.status_code == 503

    def test_returns_503_when_celery_absent(self, api_client) -> None:
        with patch("api.main.HAS_CELERY", False), \
             patch("api.main.get_task_result", None):
            resp = api_client.get("/tasks/any-id", headers=_ANALYST_HEADERS)
        assert resp.status_code == 503

    def test_requires_auth(self, api_client) -> None:
        resp = api_client.get("/tasks/any-id")
        assert resp.status_code == 401


# ===========================================================================
# I — 503 error paths
# ===========================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestInfrastructure503Paths:
    def test_graph_collusion_rings_503_when_postgres_down(self, api_client) -> None:
        from src.database import InfrastructureError
        with patch("api.main._HAS_GRAPH", True), \
             patch("api.main.GraphCollusionDetector", MagicMock()), \
             patch("api.main.assert_postgres_reachable",
                   side_effect=InfrastructureError("pg down")):
            resp = api_client.get("/graph/collusion-rings", headers=_ANALYST_HEADERS)
        assert resp.status_code == 503
        assert "pg down" in resp.json()["detail"]

    def test_graph_collusion_rings_503_when_module_absent(self, api_client) -> None:
        with patch("api.main._HAS_GRAPH", False), \
             patch("api.main.GraphCollusionDetector", None):
            resp = api_client.get("/graph/collusion-rings", headers=_ANALYST_HEADERS)
        assert resp.status_code == 503

    def test_compliance_503_when_module_absent(self, api_client) -> None:
        with patch("api.main._HAS_COMPLIANCE", False), \
             patch("api.main.generate_compliance_report", None):
            resp = api_client.get("/compliance", headers=_VIEWER_HEADERS)
        assert resp.status_code == 503

    def test_copilot_503_when_module_absent(self, api_client) -> None:
        with patch("api.main._HAS_COPILOT", False), \
             patch("api.main.run_copilot", None):
            resp = api_client.post(
                "/copilot/decide",
                json={"query": "test", "context_type": "underwriting", "features": {}},
                headers=_ANALYST_HEADERS,
            )
        assert resp.status_code == 503


# ===========================================================================
# J — /health
# ===========================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi not installed")
class TestHealthEndpoint:
    def test_health_returns_200_no_auth(self, api_client) -> None:
        resp = api_client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
        assert "timestamp" in body

    def test_health_returns_200_even_when_redis_down(self, api_client) -> None:
        """Health probe must never depend on Redis."""
        with patch("api.main.assert_redis_reachable",
                   side_effect=Exception("redis totally gone")):
            resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_version_is_string(self, api_client) -> None:
        resp = api_client.get("/health")
        assert isinstance(resp.json()["version"], str)
