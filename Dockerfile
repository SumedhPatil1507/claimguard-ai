# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /root/.local /root/.local

# Copy application source
COPY . .

# Ensure user-installed scripts are on PATH and the package root is importable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Pre-generate synthetic datasets so they are available at container startup
RUN python data/synthetic_generator.py

# FastAPI (8000)
EXPOSE 8000

# Default: run the FastAPI backend.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
