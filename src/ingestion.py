"""
ingestion.py
============
Dual Ingestion Engine for ClaimGuard AI (v1.9).

Two ingestion paths:

1. **CSV batch ingestion** — ``parse_csv`` + ``classify_csv`` normalise
   uploaded claims / policy CSVs so they can be enqueued to Celery workers
   (``src.worker.ingest_csv_batch``) for parallel batch scoring.

2. **Document ingestion (PDF / Image / Markdown)** — ``extract_text`` performs
   PDF text extraction (pypdf) or OCR (pytesseract, optional), then
   ``split_markdown`` chunks the document using Markdown H1–H3 heading
   splitters, and ``ingest_document`` upserts the chunks directly into the
   Qdrant hybrid vector store (dense + BM25) via ``policy_store``.

Every stage emits OpenTelemetry spans when the OTel SDK is configured:
    ingestion.extract / ingestion.chunk / ingestion.upsert

Graceful degradation mirrors ``src.vector_store``: a missing optional
dependency (pypdf, pytesseract, pandas) raises a typed ``IngestionError``
with an actionable message — never an opaque ImportError.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: OpenTelemetry instrumentation
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OPENTELEMETRY = True
    _tracer = trace.get_tracer(__name__)
except ImportError:
    HAS_OPENTELEMETRY = False
    _tracer = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from pypdf import PdfReader  # type: ignore
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False
    PdfReader = None  # type: ignore[assignment,misc]

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    pytesseract = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Errors & constants
# ---------------------------------------------------------------------------

class IngestionError(Exception):
    """Typed ingestion failure with an HTTP-friendly ``status_code``."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


CSV_EXTENSIONS = {".csv"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".txt"}
ALL_EXTENSIONS = CSV_EXTENSIONS | PDF_EXTENSIONS | IMAGE_EXTENSIONS | MARKDOWN_EXTENSIONS

MAX_FILE_BYTES = 25 * 1024 * 1024   # 25 MB hard cap per upload
MAX_CSV_ROWS = 10_000               # rows per CSV batch

# Column signatures used to classify an uploaded CSV.
_CLAIMS_REQUIRED = {"claim_id", "claim_amount"}
_POLICIES_REQUIRED = {"policy_id", "sum_insured"}

# Markdown H1–H3 heading regex.


# ---------------------------------------------------------------------------
# Stage 1 — file type detection
# ---------------------------------------------------------------------------

def detect_kind(filename: str, content_type: Optional[str] = None) -> str:
    """
    Classify an upload as ``csv``, ``pdf``, ``image``, ``markdown`` or
    ``unsupported`` based on extension (falling back to content-type).
    """
    name = (filename or "").lower()
    dot = name.rfind(".")
    ext = name[dot:] if dot >= 0 else ""
    if ext in CSV_EXTENSIONS:
        return "csv"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in MARKDOWN_EXTENSIONS:
        return "markdown"
    if content_type:
        ct = content_type.lower()
        if "csv" in ct:
            return "csv"
        if "pdf" in ct:
            return "pdf"
        if ct.startswith("image/"):
            return "image"
        if ct.startswith("text/"):
            return "markdown"
    return "unsupported"


def _slugify(text: str, max_len: int = 48) -> str:
    """URL/ID-safe slug for chunk IDs."""
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text[:max_len] or "section"


# ---------------------------------------------------------------------------
# Stage 2 — text extraction (PDF text layer / OCR / plain text)
# ---------------------------------------------------------------------------

def extract_text(filename: str, data: bytes, kind: str) -> str:
    """
    Extract raw text from uploaded bytes.

    * ``pdf``      — pypdf text layer extraction.
    * ``image``    — pytesseract OCR over Pillow-decoded raster.
    * ``markdown`` — UTF-8 decode (fallback latin-1).
    """
    if not data:
        raise IngestionError("Uploaded file is empty.", status_code=400)

    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("ingestion.extract") as span:
            span.set_attribute("file.name", (filename or "")[:120])
            span.set_attribute("file.kind", kind)
            span.set_attribute("file.bytes", len(data))
            return _extract_text_impl(filename, data, kind, span)
    return _extract_text_impl(filename, data, kind, None)


def _extract_text_impl(filename: str, data: bytes, kind: str, span: Any) -> str:
    try:
        if kind == "pdf":
            text = _extract_pdf(filename, data, span)
        elif kind == "image":
            text = _extract_image_ocr(filename, data, span)
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                text = data.decode("latin-1", errors="replace")
            if span:
                span.set_attribute("extract.chars", len(text))
        return text
    except IngestionError as exc:
        if span:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
        raise



def _extract_pdf(filename: str, data: bytes, span: Any) -> str:
    if not HAS_PYPDF:
        raise IngestionError(
            "PDF extraction requires the 'pypdf' package. "
            "Install it with: pip install pypdf",
            status_code=501,
        )
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [(page.extract_text() or "") for page in reader.pages]
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Could not parse PDF '{filename}': {exc}", status_code=422
        ) from exc
    text = "\n\n".join(p.strip() for p in pages if p and p.strip())
    if not text.strip():
        raise IngestionError(
            f"PDF '{filename}' yielded no extractable text "
            "(it may be a scanned image without a text layer).",
            status_code=422,
        )
    if span:
        span.set_attribute("extract.pages", len(pages))
        span.set_attribute("extract.chars", len(text))
    return text


def _extract_image_ocr(filename: str, data: bytes, span: Any) -> str:
    if not HAS_OCR:
        raise IngestionError(
            "OCR requires 'pytesseract' + Pillow (+ the Tesseract binary). "
            "Install with: pip install pytesseract pillow",
            status_code=501,
        )
    try:
        img = Image.open(io.BytesIO(data))
        text = pytesseract.image_to_string(img)
    except Exception as exc:
        raise IngestionError(
            f"OCR failed for '{filename}': {exc}", status_code=422
        ) from exc
    if not text.strip():
        raise IngestionError(f"OCR produced no text from '{filename}'.", status_code=422)
    if span:
        span.set_attribute("extract.chars", len(text))
    return text



# ---------------------------------------------------------------------------
# Stage 3 — Markdown H1–H3 heading splitter
# ---------------------------------------------------------------------------

def split_markdown(
    text: str,
    source: str,
    max_chunk_chars: int = 1500,
) -> List[Dict[str, str]]:
    """
    Split *text* on Markdown H1–H3 headings (``#``, ``##``, ``###``).

    Oversized sections are sub-split on paragraph boundaries so no chunk
    exceeds ``max_chunk_chars``.  Every chunk carries:

    * ``content`` — the heading context + section body
    * ``source``  — original file name
    * ``section`` — heading path (e.g. ``Claims Procedure > Intimation``)
    * ``chunk_id``— deterministic ``{slug}::h{n}::p{m}`` identifier used for
                     Guardrails citation enforcement.
    """
    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("ingestion.chunk") as span:
            span.set_attribute("source", source[:120])
            span.set_attribute("raw_chars", len(text))
            chunks = _split_markdown_impl(text, source, max_chunk_chars, span)
            span.set_attribute("chunks", len(chunks))
            return chunks
    return _split_markdown_impl(text, source, max_chunk_chars, None)


def _split_markdown_impl(
    text: str,
    source: str,
    max_chunk_chars: int,
    span: Any,
) -> List[Dict[str, str]]:
    matches = list(_HEADING_RE.finditer(text))
    chunks: List[Dict[str, str]] = []

    # Preamble before the first heading → its own chunk (title = file stem).
    preamble_end = matches[0].start() if matches else len(text)
    preamble = text[:preamble_end].strip()
    heading_path: List[str] = []

    def _emit(body: str, path: List[str], idx: int) -> None:
        body = body.strip()
        if not body:
            return
        section = " > ".join(path) if path else (source or "Document")
        heading_line = f"# {section}\n\n" if path else ""
        content = heading_line + body
        while len(content) > max_chunk_chars and len(body) > 100:
            # Hard sub-split at paragraph boundary.
            cut = body.rfind("\n\n", 100, max_chunk_chars)
            if cut <= 100:
                cut = max_chunk_chars
            part = body[:cut].strip()
            body = body[cut:].lstrip()
            if part:
                chunks.append({
                    "content": (heading_line + part)[:max_chunk_chars * 2],
                    "source": source,
                    "section": section,
                    "chunk_id": f"{_slugify(source)}::{_slugify(section, 40)}::p{idx}",
                })
            idx += 1
        if body:
            chunks.append({
                "content": (heading_line + body)[: max_chunk_chars * 2],
                "source": source,
                "section": section,
                "chunk_id": f"{_slugify(source)}::{_slugify(section, 40)}::p{idx}",
            })

    if preamble:
        _emit(preamble, [], 0)

    for i, match in enumerate(matches):
        level = len(match.group(1))
        title = match.group(2).strip()
        # Trim heading path to the current level (H1 resets, H2 pops one…)
        heading_path = heading_path[: level - 1]
        heading_path.append(title)
        section_start = match.end()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        _emit(text[section_start:section_end], list(heading_path), i + 1)

    if span:
        span.set_attribute("sections", len(matches))
    return chunks


# ---------------------------------------------------------------------------
# Stage 4 — Qdrant hybrid upsert
# ---------------------------------------------------------------------------

def upsert_chunks(chunks: List[Dict[str, str]]) -> int:
    """
    Upsert chunks into the Qdrant hybrid vector store (dense + BM25).

    Returns the number of chunks upserted.  Never raises on transient
    vector-store failures — logs and returns 0 so ingestion of one document
    cannot take down the API.
    """
    if not chunks:
        return 0

    def _do_upsert(span: Any) -> int:
        try:
            from src.vector_store import policy_store
            added = policy_store.add_documents(chunks)
            if span:
                span.set_attribute("upserted", added)
            return added
        except Exception as exc:
            logger.error("ingestion: Qdrant upsert failed — %s", exc, exc_info=True)
            if span:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            return 0

    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("ingestion.upsert") as span:
            span.set_attribute("chunks", len(chunks))
            return _do_upsert(span)
    return _do_upsert(None)




# ---------------------------------------------------------------------------
# Document ingestion orchestration (PDF / image / markdown → Qdrant)
# ---------------------------------------------------------------------------

def ingest_document(
    filename: str,
    data: bytes,
    content_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full document ingestion pipeline:

    detect → extract (PDF/OCR) → Markdown H1–H3 split → Qdrant hybrid upsert.

    Returns a JSON-safe summary:
    ``{filename, kind, chars, chunks, chunk_ids, source, status}``.
    """
    kind = detect_kind(filename, content_type)
    if kind in ("unsupported", "csv"):
        raise IngestionError(
            f"ingest_document expects a PDF/image/markdown upload; got '{filename}' "
            f"(detected kind: {kind}).",
            status_code=400,
        )
    if len(data) > MAX_FILE_BYTES:
        raise IngestionError(
            f"File exceeds the {MAX_FILE_BYTES // (1024 * 1024)} MB ingestion cap.",
            status_code=413,
        )

    text = extract_text(filename, data, kind)
    chunks = split_markdown(text, source=filename or "document")
    if not chunks:
        raise IngestionError(
            f"No chunkable content found in '{filename}'.", status_code=422
        )
    upserted = upsert_chunks(chunks)

    return {
        "status": "INGESTED" if upserted else "EXTRACTED",
        "filename": filename,
        "kind": kind,
        "chars": len(text),
        "chunks": len(chunks),
        "upserted": upserted,
        "sections": sorted({c["section"] for c in chunks}),
        "chunk_ids": [c["chunk_id"] for c in chunks[:50]],
        "vector_store": "qdrant_hybrid" if upserted else "unavailable",
    }


# ---------------------------------------------------------------------------
# CSV batch ingestion helpers (enqueued to Celery)
# ---------------------------------------------------------------------------

def parse_csv(data: bytes) -> List[Dict[str, Any]]:
    """
    Parse an uploaded CSV into a list of JSON-safe row dicts.

    Raises ``IngestionError`` when pandas is unavailable, the file is not
    parseable, or it exceeds ``MAX_CSV_ROWS``.
    """
    if not HAS_PANDAS:
        raise IngestionError(
            "CSV ingestion requires pandas. Install with: pip install pandas",
            status_code=501,
        )
    try:
        import pandas as pd_mod
        df = pd_mod.read_csv(io.BytesIO(data))
    except Exception as exc:
        raise IngestionError(f"Could not parse CSV: {exc}", status_code=422) from exc

    if df.empty:
        raise IngestionError("CSV contains no data rows.", status_code=422)
    if len(df) > MAX_CSV_ROWS:
        raise IngestionError(
            f"CSV has {len(df)} rows; maximum per batch is {MAX_CSV_ROWS}.",
            status_code=413,
        )

    # Normalise NaN → None and timestamps → ISO strings for JSON safety.
    df = df.where(df.notna(), None)
    rows: List[Dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        clean: Dict[str, Any] = {}
        for key, value in record.items():
            if hasattr(value, "item"):        # numpy scalar → python scalar
                try:
                    value = value.item()
                except Exception:
                    value = str(value)
            if hasattr(value, "isoformat"):   # datetime/date → ISO 8601
                value = value.isoformat()
            clean[str(key)] = value
        rows.append(clean)
    return rows


def classify_csv(columns: List[str]) -> str:
    """
    Classify a CSV by its column headers:

    * ``claims``   — has ``claim_id`` + ``claim_amount``
    * ``policies`` — has ``policy_id`` + ``sum_insured``
    * ``generic``  — anything else (ingested for archive / exploration only)
    """
    cols = {c.strip().lower() for c in columns}
    if _CLAIMS_REQUIRED <= cols:
        return "claims"
    if _POLICIES_REQUIRED <= cols:
        return "policies"
    return "generic"


def summarise_upload(filename: str, kind: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the 202 response payload for a CSV upload."""
    columns = sorted({k for row in rows[:50] for k in row})
    dataset = classify_csv(columns)
    return {
        "status": "ENQUEUED",
        "filename": filename,
        "kind": kind,
        "dataset": dataset,
        "rows": len(rows),
        "columns": columns,
    }

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
