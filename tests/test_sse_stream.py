"""
Tests for Server-Sent Events (SSE) streaming endpoints in ClaimGuard AI.
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import app, _create_access_token
from src.worker import get_task_channel, publish_task_event


from sse_starlette.sse import AppStatus


@pytest.fixture(autouse=True)
def reset_sse_event_loop():
    AppStatus.should_exit_event = None
    yield
    AppStatus.should_exit_event = None


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def auth_headers():
    token = _create_access_token(data={"sub": "admin", "role": "admin"})
    return {"Authorization": f"Bearer {token}"}


def test_task_channel_name():
    ch = get_task_channel("test-123")
    assert ch == "claimguard:task_events:test-123"


def test_publish_task_event_safe_without_crash():
    # Should not raise even if Redis connection is mocked / absent
    publish_task_event("test-123", {"status": "STARTED"})
    publish_task_event("", {"status": "STARTED"})


def test_underwrite_stream_requires_auth(client):
    res = client.get("/underwrite/stream/fake-task-id")
    assert res.status_code == 401


def test_claims_stream_requires_auth(client):
    res = client.get("/claims/stream/fake-task-id")
    assert res.status_code == 401


def test_generic_task_stream_requires_auth(client):
    res = client.get("/tasks/stream/fake-task-id")
    assert res.status_code == 401


def test_stream_with_query_token(client):
    token = _create_access_token(data={"sub": "admin", "role": "admin"})
    with patch("api.main.get_task_result", return_value={"task_id": "fake-1", "status": "SUCCESS", "result": {}}):
        res = client.get(f"/underwrite/stream/fake-1?token={token}")
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        assert "SUCCESS" in res.text


def test_stream_underwrite_success_event(client, auth_headers):
    with patch("api.main.get_task_result", return_value={"task_id": "fake-2", "status": "SUCCESS", "result": {"risk_tier": "low"}}):
        res = client.get("/underwrite/stream/fake-2", headers=auth_headers)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        assert "risk_tier" in res.text


def test_stream_claims_failure_event(client, auth_headers):
    with patch("api.main.get_task_result", return_value={"task_id": "fake-3", "status": "FAILURE", "error": "Model computation failed"}):
        res = client.get("/claims/stream/fake-3", headers=auth_headers)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers.get("content-type", "")
        assert "Model computation failed" in res.text
