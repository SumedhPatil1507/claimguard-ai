"""
src/ingestion.py
================
Dual Ingestion Engine — file-upload pipeline for ClaimGuard AI.

Supports the ``POST /ingest/file`` FastAPI endpoint:

* **CSV files**  → batch claims / policy rows validated and enqueued to the
  existing Celery scoring workers (``claimguard.score_claim`` and
  ``claimguard.score_underwriting``), one child task per row, all publishing
  live progress to the parent task's Redis Pub/Sub SSE channel.

* **PDF / image files** (policy wording scans, repair invoices, medical
  receipts) → text extraction via pypdf/pdfminer with an optional
  pytesseract OCR pass for scanned images, then split with the Markdown
  H1–H3 heading splitter and upserted directly into the Qdrant hybrid
  vector store so the Policy Copilot can cite them immediately.

Every stage emits SSE progress events through ``publish_task_event`` so the
Next.js dashboard renders a live streaming progress bar while the Celery
worker processes the upload.

Graceful degradation
--------------------
The module always imports.  Missing optional dependencies (pypdf, PIL,
pytesseract, celery) disable individual capabilities but never crash the
API process; endpoints surface a precise 415/503 instead.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------
try:
    from src.vector_store import policy_store, split_markdown_h1_h3
    HAS_VECTOR_STORE = True
except Exception as exc:  # pragma: no cover
    logger.warning("ingestion: vector store unavailable (%s)", exc)
    policy_store = None            # type: ignore[assignment]
    split_markdown_h1_h3 = None    # type: ignore[assignment]
    HAS_VECTOR_STORE = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False

try:
    from pdfminer.high_level import extract_text as _pdfminer_extract
    HAS_PDFMINER = True
except ImportError:
    HAS_PDFMINER = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

from src.worker import (
    HAS_CELERY,
    celery_app,
    publish_task_event,
    score_claim_task,
    score_underwriting_task,
)

try:
    from src.otel_config import get_tracer
    _tracer = get_tracer("claimguard.ingestion")
except Exception:  # pragma: no cover - defensive
    class _T:
        class _C:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def set_attribute(self, k, v): ...
            def set_attributes(self, a): ...
            def record_exception(self, e, attributes=None): ...
            def add_event(self, n, attributes=None): ...
            def set_status(self, s=None, d=None): ...
        def start_as_current_span(self, name, **kw): return self._C()
    _tracer = _T()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# File-type classification
# ---------------------------------------------------------------------------

CSV_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
TEXT_EXTENSIONS = {".txt", ".md"}

MAX_FILE_BYTES = int(os.environ.get("INGEST_MAX_FILE_MB", "25")) * 1024 * 1024


def classify_file(filename: str, content_type: str = "") -> str:
    """Return one of: csv | pdf | image | text | unsupported."""
    ext = Path(filename).suffix.lower()
    ct = (content_type or "").lower()
    if ext in CSV_EXTENSIONS or "csv" in ct or "excel" in ct:
        return "csv"
    if ext in PDF_EXTENSIONS or "pdf" in ct:
        return "pdf"
    if ext in IMAGE_EXTENSIONS or ct.startswith("image/"):
        return "image"
    if ext in TEXT_EXTENSIONS or ct.startswith("text/"):
        return "text"
    return "unsupported"


# ---------------------------------------------------------------------------
# Text extraction (PDF / image / plain text)
# ---------------------------------------------------------------------------

def extract_pdf_text(data: bytes) -> str:
    """Extract text layer from a PDF; fall back to per-page OCR when empty."""
    parts: List[str] = []

    if HAS_PYPDF:
        try:
            reader = PdfReader(io.BytesIO(data))
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception as exc:
                    logger.debug("ingestion: pypdf page %d failed: %s", i, exc)
                    txt = ""
                if txt.strip():
                    parts.append(f"# Page {i + 1}\n{txt}")
        except Exception as exc:
            logger.warning("ingestion: pypdf extraction failed: %s", exc)

    if not any(p.strip() for p in parts) and HAS_PDFMINER:
        try:
            raw = _pdfminer_extract(io.BytesIO(data)) or ""
            if raw.strip():
                parts.append(raw)
        except Exception as exc:
            logger.warning("ingestion: pdfminer extraction failed: %s", exc)

    # Scanned PDF with no text layer → rasterise pages and OCR them.
    if not any(len(p.strip()) > 40 for p in parts) and HAS_TESSERACT and HAS_PIL:
        try:
            import fitz  # PyMuPDF — optional

            doc = fitz.open(stream=data, filetype="pdf")
            ocr_pages: List[str] = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                txt = pytesseract.image_to_string(img)
                if txt.strip():
                    ocr_pages.append(f"# Page {i + 1} (OCR)\n{txt}")
            if ocr_pages:
                parts = ocr_pages
        except Exception as exc:
            logger.info("ingestion: scanned-PDF OCR unavailable (%s)", exc)

    return "\n\n".join(parts)


def extract_image_text(data: bytes, filename: str) -> str:
    """OCR an uploaded invoice / medical receipt / policy scan."""
    if not (HAS_TESSERACT and HAS_PIL):
        raise RuntimeError(
            "OCR requires the optional packages 'pytesseract' and Pillow "
            "(and a system tesseract-ocr binary). Install with: "
            "pip install pytesseract pillow && apt-get install tesseract-ocr"
        )
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    text = pytesseract.image_to_string(img)
    header = f"# Document: {filename}"
    return f"{header}\n\n{text}" if text.strip() else header


def extract_text_content(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


def normalize_to_markdown(raw_text: str, source: str) -> str:
    """
    Best-effort promotion of extracted text into Markdown heading structure.

    Insurance documents rarely arrive as literal Markdown, so we detect
    common clause/section numbering conventions ("SECTION 4.2 — ...",
    "4.2 Non-Disclosure", "Clause 7.3:") and rewrite them as H2/H3 headings
    before the H1–H3 splitter runs.  This keeps every ingested chunk
    anchored to a citable structural heading.
    """
    import re

    lines = raw_text.splitlines()
    out: List[str] = [f"# {Path(source).stem.replace('_', ' ').title()}"]
    pattern_major = re.compile(r"^\s*(?:SECTION|ARTICLE|PART)\s+([0-9]+(?:\.[0-9]+)?)\b[\s—:-]*(.*)$", re.I)
    pattern_clause = re.compile(r"^\s*(?:CLAUSE\s+)?(\d+\.\d+)\b[\s.—:-]+(.+)$")
    for line in lines:
        m = pattern_major.match(line)
        if m:
            out.append(f"## Section {m.group(1)} — {m.group(2).strip() or 'Untitled'}")
            continue
        m = pattern_clause.match(line)
        if m:
            out.append(f"### Clause {m.group(1)} {m.group(2).strip()}")
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Document ingestion → Qdrant hybrid vector store
# ---------------------------------------------------------------------------

def ingest_document_bytes(data: bytes, filename: str, kind: str) -> Dict[str, Any]:
    """
    Extract, normalise, Markdown H1–H3 chunk and upsert one document into
    the Qdrant hybrid store.  Returns an ingestion summary dict.
    """
    with _tracer.start_as_current_span("ingestion.document") as sp:
        sp.set_attribute("file.name", filename)
        sp.set_attribute("file.kind", kind)
        sp.set_attribute("file.bytes", len(data))

        if kind == "pdf":
            raw = extract_pdf_text(data)
        elif kind == "image":
            raw = extract_image_text(data, filename)
        else:
            raw = extract_text_content(data)

        if not raw.strip():
            return {"status": "empty", "chunks": 0, "source": filename,
                    "preview": "", "chunk_ids": []}

        md = normalize_to_markdown(raw, filename)
        segments = (
            split_markdown_h1_h3(md) if HAS_VECTOR_STORE and split_markdown_h1_h3
            else [{"heading": "(document)", "content": md}]
        )

        chunks: List[Dict[str, str]] = []
        for seg in segments:
            body = seg["content"]
            heading = seg.get("heading", "")
            content = f"## {heading}\n{body}" if heading and not body.lstrip().startswith("#") else body
            chunks.append({"content": content, "source": filename})

        added = 0
        chunk_ids: List[str] = []
        if HAS_VECTOR_STORE and policy_store is not None:
            try:
                added = policy_store.add_chunks(chunks)
                # Rebuild deterministic ids for the response payload.
                import hashlib as _h
                import re as _re
                for c in chunks:
                    digest = _h.sha1(c["content"].encode()).hexdigest()[:12]
                    slug = _re.sub(r"[^a-zA-Z0-9]+", "_", Path(filename).stem).strip("_").lower()[:40]
                    chunk_ids.append(f"{slug or 'doc'}_{digest}")
            except Exception as exc:
                logger.error("ingestion: vector upsert failed — %s", exc)
                raise

        sp.set_attribute("chunks_upserted", added)
        logger.info("ingestion: %s → %d markdown chunks upserted into Qdrant", filename, added)
        return {
            "status": "indexed",
            "source": filename,
            "kind": kind,
            "chunks": added,
            "characters": len(raw),
            "chunk_ids": chunk_ids[:10],
            "preview": chunks[0]["content"][:300] if chunks else "",
        }


# ---------------------------------------------------------------------------
# CSV batch ingestion → Celery fan-out
# ---------------------------------------------------------------------------

_UW_REQUIRED = {"policy_id", "age", "sum_insured"}
_CLAIM_REQUIRED = {"claim_id", "claim_amount"}


def _row_to_features(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce CSV cell values to JSON-safe scalars."""
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, float) and (value != value):  # NaN
            continue
        try:
            json.dumps(value)
            out[str(key)] = value
        except TypeError:
            out[str(key)] = str(value)
    return out


def ingest_csv_bytes(data: bytes, filename: str, batch_id: str,
                    parent_task_id: str = "") -> Dict[str, Any]:
    """
    Parse a claims/policies CSV, auto-detect its schema, and enqueue one
    Celery scoring job per row.  Live progress is published to the parent
    task's SSE channel after every batch of rows.
    """
    with _tracer.start_as_current_span("ingestion.csv_batch") as sp:
        sp.set_attribute("file.name", filename)
        if not HAS_PANDAS:
            raise RuntimeError("CSV ingestion requires pandas.")

        df = pd.read_csv(io.BytesIO(data))
        columns = {c.lower().strip(): c for c in df.columns}
        sp.set_attribute("csv.rows", len(df))

        if all(k in columns for k in _CLAIM_REQUIRED):
            mode = "claims"
        elif all(k in columns for k in _UW_REQUIRED):
            mode = "underwriting"
        else:
            return {
                "status": "rejected",
                "reason": (
                    "CSV schema unrecognised. Claims CSVs must contain at least "
                    f"{sorted(_CLAIM_REQUIRED)}; policy CSVs at least {sorted(_UW_REQUIRED)}. "
                    f"Found: {list(df.columns)[:12]}"
                ),
                "rows": int(len(df)),
            }

        task_fn = score_claim_task if mode == "claims" else score_underwriting_task
        enqueued = 0
        child_ids: List[str] = []
        errors: List[str] = []

        for _, series in df.iterrows():
            features = _row_to_features(series.to_dict())
            child_id = str(uuid.uuid4())
            try:
                if not (HAS_CELERY and task_fn is not None):
                    raise RuntimeError("Celery worker stack unavailable.")
                task_fn.apply_async(kwargs={"features": features}, task_id=child_id)
                enqueued += 1
                child_ids.append(child_id)
            except Exception as exc:
                errors.append(str(exc)[:200])
                if len(errors) >= 3:
                    break

            if parent_task_id and enqueued % 10 == 0:
                total = max(len(df), 1)
                publish_task_event(parent_task_id, {
                    "task_id": parent_task_id,
                    "status": "PROGRESS",
                    "progress": {
                        "stage": f"Enqueued {enqueued}/{total} {mode} jobs",
                        "percent": round(10 + 80 * enqueued / total, 1),
                        "batch_id": batch_id,
                        "mode": mode,
                    },
                })

        if parent_task_id:
            publish_task_event(parent_task_id, {
                "task_id": parent_task_id,
                "status": "PROGRESS",
                "progress": {
                    "stage": f"Batch complete — {enqueued} {mode} jobs dispatched to workers",
                    "percent": 95,
                    "batch_id": batch_id,
                    "mode": mode,
                },
            })

        sp.set_attribute("jobs_enqueued", enqueued)
        return {
            "status": "dispatched",
            "mode": mode,
            "rows": int(len(df)),
            "jobs_enqueued": enqueued,
            "child_task_ids": child_ids[:50],
            "errors": errors[:5],
            "batch_id": batch_id,
        }


# ---------------------------------------------------------------------------
# Celery task wrapper — registered only when Celery is available
# ---------------------------------------------------------------------------

if HAS_CELERY and celery_app is not None:

    @celery_app.task(name="claimguard.ingest_file", bind=True, max_retries=2)
    def ingest_file_task(self, *, path: str, filename: str, kind: str,
                         parent_task_id: str = "") -> Dict[str, Any]:
        """
        Background ingestion of an uploaded file.

        ``path`` is a local temp-file written by the API layer before the
        task was enqueued (shared volume in Docker deployments).
        """
        task_id = self.request.id or parent_task_id
        publish_task_event(task_id, {
            "task_id": task_id, "status": "STARTED",
            "progress": {"stage": f"Processing upload {filename}", "percent": 10,
                         "kind": kind},
        })
        try:
            data = Path(path).read_bytes()
            if kind == "csv":
                result = ingest_csv_bytes(data, filename, batch_id=task_id,
                                          parent_task_id=task_id)
            else:
                result = ingest_document_bytes(data, filename, kind)
            result["task_id"] = task_id
            publish_task_event(task_id, {
                "task_id": task_id, "status": "SUCCESS", "result": result,
                "progress": {"stage": "Ingestion complete", "percent": 100},
            })
            return result
        except Exception as exc:
            publish_task_event(task_id, {
                "task_id": task_id, "status": "FAILURE", "error": str(exc),
                "progress": {"stage": "Ingestion failed", "percent": 100},
            })
            raise
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

else:
    def ingest_file_task(*args: Any, **kwargs: Any) -> None:  # type: ignore[misc]
        raise RuntimeError("Celery is not installed; file ingestion is unavailable.")


# ---------------------------------------------------------------------------
# Temp-file helper used by the API layer
# ---------------------------------------------------------------------------

UPLOAD_DIR = Path(tempfile.gettempdir()) / "claimguard_uploads"


def save_upload_tmp(data: bytes, filename: str) -> str:
    """Persist an upload to a worker-visible temp path; returns the path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe = f"{uuid.uuid4().hex}_{Path(filename).name}"
    dest = UPLOAD_DIR / safe
    dest.write_bytes(data)
    return str(dest)


__all__ = [
    "classify_file",
    "extract_pdf_text",
    "extract_image_text",
    "normalize_to_markdown",
    "ingest_document_bytes",
    "ingest_csv_bytes",
    "ingest_file_task",
    "save_upload_tmp",
    "MAX_FILE_BYTES",
]
