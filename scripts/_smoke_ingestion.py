"""Quick smoke check for src/ingestion.py — run: python scripts/_smoke_ingestion.py"""
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import (  # noqa: E402
    IngestionError,
    classify_csv,
    detect_kind,
    extract_text,
    ingest_document,
    parse_csv,
    split_markdown,
)

MD = (
    "# ClaimGuard Standard Policy\n\n"
    "Intro paragraph about coverage.\n\n"
    "## Claims Procedure\n\n"
    "Notify the insurer within 7 days of the incident.\n\n"
    "### Intimation\n\n"
    "Call the 24x7 hotline or file online.\n\n"
    "## Exclusions\n\n"
    "Wear and tear is excluded from cover.\n"
)

chunks = split_markdown(MD, "policy_wording.md")
assert chunks, "expected chunks"
for c in chunks:
    assert set(c) == {"content", "source", "section", "chunk_id"}
    print(f"  {c['chunk_id']}  |  {c['section']}")

assert detect_kind("a.pdf") == "pdf"
assert detect_kind("b.CSV") == "csv"
assert detect_kind("c.png") == "image"
assert detect_kind("d.md") == "markdown"
assert detect_kind("e.zip") == "unsupported"
assert classify_csv(["claim_id", "claim_amount"]) == "claims"
assert classify_csv(["policy_id", "sum_insured"]) == "policies"
assert classify_csv(["foo", "bar"]) == "generic"

csv_bytes = b"claim_id,claim_amount,claim_type\nCLM-1,50000,motor\nCLM-2,75000,health\n"
rows = parse_csv(csv_bytes)
assert len(rows) == 2 and rows[0]["claim_id"] == "CLM-1"
print(f"  parse_csv -> {len(rows)} rows, dataset={classify_csv(list(rows[0]))}")

# Empty file must raise IngestionError
try:
    extract_text("x.pdf", b"", "pdf")
    raise AssertionError("expected IngestionError")
except IngestionError as exc:
    print(f"  empty-file guard OK (status={exc.status_code})")

# ingest_document must reject CSVs with 400
try:
    ingest_document("batch.csv", csv_bytes)
    raise AssertionError("expected IngestionError")
except IngestionError as exc:
    assert exc.status_code == 400
    print(f"  csv-rejected-by-document-path OK (status={exc.status_code})")

# PDF roundtrip: build a tiny PDF with fpdf2 if available, else skip.
try:
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "Motor Policy Wording. Claim notification within 7 days.")
    pdf_bytes = pdf.output()
    text = extract_text("sample.pdf", bytes(pdf_bytes), "pdf")
    assert "Claim notification" in text or "Policy" in text
    print(f"  pypdf extract OK -> {len(text)} chars")
except ImportError:
    print("  fpdf2 not available; skipping PDF roundtrip")

print("[ingestion] ALL SMOKE CHECKS PASSED")
