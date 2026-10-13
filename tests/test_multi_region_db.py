"""Unit tests for database endpoint failover and serialization retries."""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.database import DatabaseManager


class SerializationFailure(Exception):
    sqlstate = "40001"


class _Connection:
    def __init__(self):
        self.close = AsyncMock()

    @asynccontextmanager
    async def transaction(self, **_kwargs):
        yield


def test_serializable_transaction_retries_40001(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://primary/db")
    manager = DatabaseManager()
    connections = [_Connection(), _Connection()]
    manager._asyncpg_connect = AsyncMock(side_effect=connections)
    calls = 0

    async def operation(_conn):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise SerializationFailure("restart transaction")
        return "committed"

    with patch("src.database.asyncio.sleep", new_callable=AsyncMock) as sleep, \
         patch("src.database.random.uniform", return_value=1.0):
        result = asyncio.run(manager._run_serializable_transaction(operation))

    assert result == "committed"
    assert calls == 2
    assert manager._asyncpg_connect.await_count == 2
    sleep.assert_awaited_once()
    assert all(conn.close.await_count == 1 for conn in connections)


def test_serializable_transaction_does_not_retry_other_errors(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://primary/db")
    manager = DatabaseManager()
    connection = _Connection()
    manager._asyncpg_connect = AsyncMock(return_value=connection)
    operation = AsyncMock(side_effect=RuntimeError("constraint failure"))

    with pytest.raises(RuntimeError, match="constraint failure"):
        asyncio.run(manager._run_serializable_transaction(operation))

    assert manager._asyncpg_connect.await_count == 1
    connection.close.assert_awaited_once()


def test_read_endpoint_falls_back_from_replica_to_primary(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://primary/db")
    monkeypatch.setenv("DATABASE_READ_REPLICA_URLS", "postgresql://replica/db")
    manager = DatabaseManager()
    replica = MagicMock()
    replica.fetch = AsyncMock(side_effect=ConnectionError("replica unavailable"))
    replica.close = AsyncMock()
    primary = MagicMock()
    primary.fetch = AsyncMock(return_value=[{"claim_id": "c-1"}])
    primary.close = AsyncMock()

    connect = AsyncMock(side_effect=[replica, primary])
    import src.database as database_module
    monkeypatch.setattr(database_module, "HAS_ASYNCPG", True)
    monkeypatch.setattr(database_module, "asyncpg", SimpleNamespace(connect=connect), raising=False)
    with patch("src.database.asyncio.sleep", new_callable=AsyncMock):
        rows = asyncio.run(manager._asyncpg_fetch("SELECT * FROM claims"))

    assert rows == [{"claim_id": "c-1"}]
    assert [call.args[0] for call in connect.await_args_list] == [
        "postgresql://replica/db", "postgresql://primary/db",
    ]
    replica.close.assert_awaited_once()
    primary.close.assert_awaited_once()
