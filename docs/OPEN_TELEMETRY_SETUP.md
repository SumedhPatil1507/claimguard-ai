# OpenTelemetry and Guardrails AI Setup Guide

This document explains how to configure OpenTelemetry tracing and Guardrails AI validation for the ClaimGuard AI LangGraph agent.

## OpenTelemetry Configuration

OpenTelemetry is configured to export traces to an OTLP collector for distributed tracing and observability.

### Environment Variables

Add the following environment variables to your `.env` file:

```bash
# OpenTelemetry OTLP Collector Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=claimguard-ai
OTEL_TRACES_SAMPLER_ARG=1.0
```

### Configuration Details

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTEL_EXPORTER_OTLP_ENDPOINT` | No | `http://localhost:4317` | OTLP collector endpoint for trace export |
| `OTEL_SERVICE_NAME` | No | `claimguard-ai` | Service name for trace metadata |
| `OTEL_TRACES_SAMPLER_ARG` | No | `1.0` | Sampling rate (0.0 to 1.0) |

### OTLP Collector Setup

To receive and visualize traces, you need an OTLP-compatible collector. Common options:

#### Option 1: OpenTelemetry Collector

```bash
# Download and run the OpenTelemetry Collector
docker run -p 4317:4317 -p 4318:4318 \
  -v $(pwd)/otel-collector-config.yaml:/etc/otel-collector-config.yaml \
  otel/opentelemetry-collector:latest
```

Example `otel-collector-config.yaml`:
```yaml
receivers:
  otlp:
    protocols:
      grpc:
      http:

exporters:
  logging:
    loglevel: debug

service:
  pipelines:
    traces:
      receivers: [otlp]
      exporters: [logging]
```

#### Option 2: Jaeger with OTLP

```bash
docker run -d --name jaeger \
  -e COLLECTOR_OTLP_ENABLED=true \
  -p 4317:4317 \
  -p 4318:4318 \
  -p 16686:16686 \
  jaegertracing/all-in-one:latest
```

Access Jaeger UI at: http://localhost:16686

#### Option 3: Grafana Tempo

```bash
docker run -d --name tempo \
  -p 4317:4317 \
  -p 3200:3200 \
  grafana/tempo:latest
```

Access Tempo UI at: http://localhost:3200

### Trace Spans

The following operations are instrumented with OpenTelemetry spans:

#### Vector Store (`src/vector_store.py`)
- `vector_store.search` - Main search operation
- `vector_store.dense_search` - Qdrant dense vector search
- `vector_store.bm25_search` - BM25 sparse search
- `vector_store.rrf_fusion` - Reciprocal Rank Fusion
- `vector_store.rerank` - Re-ranking orchestration
- `vector_store.rerank_cross_encoder` - Cross-encoder re-ranking
- `vector_store.rerank_cohere` - Cohere API re-ranking

#### Agent Graph (`src/agent_graph.py`)
- `agent_graph.run_copilot` - Complete copilot pipeline execution
- `agent_graph.retriever_node` - Document retrieval node
- `agent_graph.tool_node` - ML scoring tool node
- `agent_graph.writer_node` - Decision draft generation with Guardrails validation
- `agent_graph.hitl_router_node` - Human-in-the-loop routing

Each span includes relevant attributes:
- Query text (truncated)
- Session ID
- Context type (underwriting/claims)
- Result counts
- Error status (if applicable)
- Model names and configurations

## Guardrails AI Configuration

Guardrails AI validates Policy Copilot outputs against IRDAI compliance requirements.

### Installation

The package is already included in `requirements.txt`:
```
guardrails-ai>=0.4.0
```

### Validation Rules

The IRDAI compliance validator enforces:

1. **Mandatory Human Review**: All decision drafts must explicitly state they require human analyst review per IRDAI guidelines
2. **No Unverified Approvals**: The system refuses to emit unverified coverage approvals
3. **Regulatory Citations**: Drafts should cite specific IRDAI regulations or policy clauses when applicable
4. **Professional Language**: Output must maintain professional, regulatory-compliant tone
5. **Clear Review Requirement**: No final approval or denial can be issued without human review

### Implementation

The validator is integrated in `src/agent_graph.py` in the `writer_node` function:

```python
if HAS_GUARDRAILS and get_irdai_validator is not None:
    validator = get_irdai_validator()
    validation_result = validator.validate_draft(draft)

    if not validation_result["is_valid"]:
        # Append compliance warning to the draft
        draft = f"{draft}\n\n[COMPLIANCE NOTE: This draft failed IRDAI compliance validation...]"
```

### Fallback Behavior

If Guardrails AI is not installed, the system falls back to keyword-based validation:
- Checks for "human review" or "human analyst" keywords
- Detects unverified approval keywords
- Validates regulatory reference keywords

### Configuration

No additional environment variables are required for Guardrails AI. The validator is automatically initialized when the package is available.

## Complete .env Example

```bash
# ClaimGuard AI Environment Configuration

# ── OpenTelemetry OTLP Collector ─────────────────────────────────────────────
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=claimguard-ai
OTEL_TRACES_SAMPLER_ARG=1.0

# ── Groq API (for LangGraph writer_node LLM) ─────────────────────────────────
GROQ_API_KEY=your_groq_api_key_here

# ── Vector Store Configuration ───────────────────────────────────────────────
CLAIMGUARD_VS_QDRANT_MODE=local
CLAIMGUARD_VS_QDRANT_URL=http://localhost:6333
CLAIMGUARD_VS_QDRANT_API_KEY=
CLAIMGUARD_VS_COHERE_API_KEY=

# ── Celery & Redis ───────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# ── PostgreSQL ──────────────────────────────────────────────────────────────
DATABASE_URL=postgresql://user:password@localhost:5432/claimguard

# ── JWT Authentication ──────────────────────────────────────────────────────
JWT_SECRET_KEY=your_jwt_secret_key_here
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=480

# ── API Keys (JSON object mapping keys to roles) ────────────────────────────
API_KEYS={"key1":"admin","key2":"analyst"}
```

## Verification

### Verify OpenTelemetry

1. Start your OTLP collector (e.g., Jaeger)
2. Run the FastAPI application:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
3. Make a copilot request:
   ```bash
   curl -X POST http://localhost:8000/copilot/decide \
     -H "Content-Type: application/json" \
     -d '{"query":"Test query","context_type":"underwriting","features":{}}'
   ```
4. Check your OTLP collector UI for traces from `claimguard-ai`

### Verify Guardrails AI

1. Ensure `guardrails-ai` is installed:
   ```bash
   pip list | grep guardrails
   ```
2. Run a copilot request and check logs for:
   ```
   writer_node: IRDAI compliance validation passed
   ```
   or
   ```
   writer_node: IRDAI compliance validation failed for draft
   ```

## Troubleshooting

### OpenTelemetry not exporting traces

- Verify `OTEL_EXPORTER_OTLP_ENDPOINT` is correct
- Check that the OTLP collector is running and accessible
- Review application logs for OpenTelemetry initialization errors
- Try setting `OTEL_TRACES_SAMPLER_ARG=1.0` to ensure all traces are exported

### Guardrails AI validation errors

- Verify `guardrails-ai` is installed: `pip install guardrails-ai>=0.4.0`
- Check application logs for import errors
- The system will fall back to keyword-based validation if Guardrails AI is unavailable

### Missing spans in traces

- Ensure the tracer is properly initialized in `src/otel_config.py`
- Check that spans are being created with the correct parent-child relationships
- Verify that the application is not silently swallowing exceptions

## Further Reading

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [Guardrails AI Documentation](https://www.guardrailsai.com/docs)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Grafana Tempo Documentation](https://grafana.com/docs/tempo/latest/)
