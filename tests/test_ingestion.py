"""
tests/test_ingestion.py
========================
Unit tests for the Dual Ingestion Engine (`POST /ingest/file`) and the
src.ingestion helpers (type detection, Markdown H1–H3 chunking, CSV parsing).

No live Redis / Qdrant / model downloads are required: vector-store and
Celery calls are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

CSV_BYTES = (
    b"claim_id,claimant_id,claim_amount,claim_type,claim_severity\n"
    b"CLM-1,CLT-1,50000,motor,low\n"
    b"CLM-2,CLT-2,250000,health,high\n"
)

POLICY_CSV_BYTES = (
    b"policy_id,sum_insured,coverage_type\n"
    b"POL-1,1000000,comprehensive\n"
)


# ===========================================================================
# A — src.ingestion helpers
# ===========================================================================

class TestDetectKind:
    def test_extensions(self) -> None:
        from src.ingestion import detect_kind
        assert detect_kind("claims.csv") == "csv"
        assert detect_kind("policy.PDF") == "pdf"
        assert detect_kind("receipt.png") == "image"
        assert detect_kind("wording.md") == "markdown"
        assert detect_kind("archive.zip") == "unsupported"

    def test_content_type_fallback(self) -> None:
        from src.ingestion import detect_kind
        assert detect_kind("noext", "text/csv") == "csv"
        assert detect_kind("noext", "application/pdf") == "pdf"
        assert detect_kind("noext", "image/jpeg") == "image"


class TestMarkdownSplitter:
    def test_h1_h3_sections(self) -> None:
        from src.ingestion import split_markdown
        md = (
            "# Title\nIntro.\n\n"
            "## Claims Procedure\nNotify in 7 days.\n\n"
            "### Intimation\nCall the hotline.\n\n"
            "## Exclusions\nWear and tear.\n"
        )
        chunks = split_markdown(md, "policy.md")
        assert len(chunks) == 4
        sections = [c["section"] for c in chunks]
        assert "Title > Claims Procedure > Intimation" in sections
        for c in chunks:
            assert c["chunk_id"].startswith("policy_md::")
            assert c["source"] == "policy.md"

    def test_oversized_section_sub_split(self) -> None:
        from src.ingestion import split_markdown
        body = ("Lorem ipsum dolor sit amet. " * 400).strip()
        chunks = split_markdown(f"# Big\n{body}", "big.md", max_chunk_chars=500)
        assert len(chunks) > 1
        assert all(len(c["content"]) <= 1200 for c in chunks)

    def test_empty_text_returns_no_chunks(self) -> None:
        from src.ingestion import split_markdown
        assert split_markdown("   ", "x.md") == []


class TestCsvParsing:
    def test_parse_rows(self) -> None:
        from src.ingestion import parse_csv
        rows = parse_csv(CSV_BYTES)
        assert len(rows) == 2
        assert rows[0]["claim_id"] == "CLM-1"

    def test_parse_garbage_raises(self) -> None:
        from src.ingestion import IngestionError, parse_csv
        with pytest.raises(IngestionError):
            parse_csv(b"\x00\x01\x02 not,a valid csv")

    def test_empty_csv_raises(self) -> None:
        from src.ingestion import IngestionError, parse_csv
        with pytest.raises(IngestionError):
            parse_csv(b"claim_id,claim_amount\n")

    def test_classify_claims_policies_generic(self) -> None:
        from src.ingestion import classify_csv
        assert classify_csv(["claim_id", "claim_amount"]) == "claims"
        assert classify_csv(["policy_id", "sum_insured"]) == "policies"
        assert classify_csv(["a", "b"]) == "generic"


class TestIngestDocument:
    def test_rejects_csv_path(self) -> None:
        from src.ingestion import IngestionError, ingest_document
        with pytest.raises(IngestionError) as exc:
            ingest_document("batch.csv", CSV_BYTES)
        assert exc.value.status_code == 400

    def test_rejects_empty_file(self) -> None:
        from src.ingestion import IngestionError, extract_text
        with pytest.raises(IngestionError):
            extract_text("x.pdf", b"", "pdf")

    def test_markdown_pipeline_upserts(self) -> None:
        from src import ingestion
        md = b"# Wording\nSome insured clauses here.\n## Exclusions\nWater damage excluded."
        with patch.object(ingestion, "upsert_chunks", return_value=2) as mock_upsert:
            stats = ingestion.ingest_document("wording.md", md, "text/markdown")
        assert stats["status"] == "INGESTED"
        assert stats["kind"] == "markdown"
        assert stats["chunks"] == 2
        assert stats["upserted"] == 2
        mock_upsert.assert_called_once()


# ===========================================================================
# B — POST /ingest/file endpoint (FastAPI)
# ===========================================================================

@pytest.mark.skipif(not HAS_FASTAPI, reason="fastapi/testclient not installed")
class TestIngestFileEndpoint:
    @pytest.fixture()
    def client(self):
        from api.main import app
        return TestClient(app)

    @pytest.fixture()
    def auth_headers(self):
        from api.main import _create_access_token
        token = _create_access_token(data={"sub": "admin", "role": "admin"})
        return {"Authorization": f"Bearer {token}"}

    def test_requires_auth(self, client) -> None:
        res = client.post(
            "/ingest/file",
            files={"file": ("a.csv", CSV_BYTES, "text/csv")},
        )
        assert res.status_code == 401

    def test_rejects_empty_file(self, client, auth_headers) -> None:
        res = client.post(
            "/ingest/file",
            files={"file": ("a.csv", b"", "text/csv")},
            headers=auth_headers,
        )
        assert res.status_code == 400

    def test_rejects_unsupported_extension(self, client, auth_headers) -> None:
        res = client.post(
            "/ingest/file",
            files={"file": ("malware.zip", b"PK\x03\x04", "application/zip")},
            headers=auth_headers,
        )
        assert res.status_code == 400
        assert "Unsupported" in res.json()["detail"]

    def test_csv_enqueues_celery_batch(self, client, auth_headers) -> None:
        mock_task = MagicMock()
        with patch("api.main.assert_redis_reachable"), \
             patch("api.main.ingest_csv_batch", mock_task):
            res = client.post(
                "/ingest/file",
                files={"file": ("claims.csv", CSV_BYTES, "text/csv")},
                headers=auth_headers,
            )
        assert res.status_code == 202
        body = res.json()
        assert body["status"] == "ENQUEUED"
        assert body["dataset"] == "claims"
        assert body["rows"] == 2
        assert body["task_id"]
        assert body["stream_url"].endswith(body["task_id"])
        mock_task.apply_async.assert_called_once()
        kwargs = mock_task.apply_async.call_args.kwargs
        assert kwargs["task_id"] == body["task_id"]
        assert kwargs["kwargs"]["kind"] == "claims"

    def test_policy_csv_classified_as_policies(self, client, auth_headers) -> None:
        mock_task = MagicMock()
        with patch("api.main.assert_redis_reachable"), \
             patch("api.main.ingest_csv_batch", mock_task):
            res = client.post(
                "/ingest/file",
                files={"file": ("policies.csv", POLICY_CSV_BYTES, "text/csv")},
                headers=auth_headers,
            )
        assert res.status_code == 202
        assert res.json()["dataset"] == "policies"

    def test_csv_returns_503_without_redis(self, client, auth_headers) -> None:
        from src.database import InfrastructureError
        with patch(
            "api.main.assert_redis_reachable",
            side_effect=InfrastructureError("redis down"),
        ):
            res = client.post(
                "/ingest/file",
                files={"file": ("claims.csv", CSV_BYTES, "text/csv")},
                headers=auth_headers,
            )
        assert res.status_code == 503

    def test_markdown_ingests_and_returns_chunk_stats(self, client, auth_headers) -> None:
        md = b"# Policy Wording\nInsured clauses.\n## Exclusions\nFlood excluded."
        fake_stats = {
            "status": "INGESTED", "filename": "wording.md", "kind": "markdown",
            "chars": 74, "chunks": 2, "upserted": 2,
            "sections": ["Policy Wording", "Policy Wording > Exclusions"],
            "chunk_ids": ["wording_md::policy_wording::p1"],
            "vector_store": "qdrant_hybrid",
        }
        with patch("api.main.ingest_document", return_value=fake_stats) as mock_ingest:
            res = client.post(
                "/ingest/file",
                files={"file": ("wording.md", md, "text/markdown")},
                headers=auth_headers,
            )
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "INGESTED"
        assert body["chunks"] == 2
        assert body["vector_store"] == "qdrant_hybrid"
        mock_ingest.assert_called_once()

    def test_viewer_role_forbidden(self, client) -> None:
        from api.main import _create_access_token
        token = _create_access_token(data={"sub": "viewer", "role": "viewer"})
        res = client.post(
            "/ingest/file",
            files={"file": ("a.csv", CSV_BYTES, "text/csv")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 403


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

