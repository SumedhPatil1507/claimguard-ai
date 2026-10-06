<div align="center">

# 🛡️ ClaimGuard AI

### Agentic InsurTech Platform · Underwriting Risk Scoring · Claims Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Celery](https://img.shields.io/badge/Celery-5.4-37814A?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost)](https://xgboost.ai)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-00A67E?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9-DB4437?style=for-the-badge&logo=qdrant)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-149%20passing-brightgreen?style=for-the-badge&logo=pytest)](tests/)

<br/>

> **ClaimGuard AI** combines ensemble ML, agentic RAG, and graph analytics to automate and explain the two most critical insurance workflows: **underwriting risk scoring** at policy-issuance time and **claims fraud detection** at filing time — all behind a mandatory Human-in-the-Loop review gate.

<br/>

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [📡 API Reference](#-api-reference) · [🖥️ UI Tabs](#️-streamlit-ui-tabs) · [⚙️ Configuration](#️-configuration) · [🧪 Testing](#-testing)

</div>

---

## ✨ What's New

| Version | Highlights |
|---------|-----------|
| **v1.3** | 🔄 **Celery+Redis** async task queue — `/underwrite` & `/claims/score` return instant `task_id`; poll `/tasks/{id}` |
| **v1.2** | 🔍 **Hybrid vector search** — ChromaDB replaced with Qdrant + BM25 + Cross-Encoder re-ranking |
| **v1.1** | 🌐 Streamlit Cloud deployment; graceful degradation for all optional deps |
| **v1.0** | Full platform: ML ensemble, LangGraph copilot, HITL queue, IRDAI compliance, Prometheus |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ClaimGuard AI  v1.3                             │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        Streamlit UI  (8 tabs)                        │   │
│   │  Explorer │ Underwrite │ Claims │ Copilot │ Graph │ HITL │ Obs │ ✅  │   │
│   └─────────────────────────────┬───────────────────────────────────────┘   │
│                                 │  HTTP                                      │
│   ┌─────────────────────────────▼───────────────────────────────────────┐   │
│   │                   FastAPI  (X-API-Key RBAC + Rate Limit)             │   │
│   │  POST /underwrite → 202    POST /claims/score → 202                  │   │
│   │  GET  /tasks/{id}          POST /copilot/decide                      │   │
│   │  GET  /graph/collusion-rings   GET /compliance   GET /metrics        │   │
│   └───┬─────────────────┬───────────────────────────────────────────────┘   │
│       │  enqueue        │  inline                                           │
│   ┌───▼──────────┐  ┌───▼──────────────────────────────────────────────┐   │
│   │ Redis  +     │  │               Domain Modules                      │   │
│   │ Celery Worker│  │  UnderwritingEngine  │  FraudDetectionEngine       │   │
│   │              │  │  (XGBoost+LightGBM  │  (XGBoost+LightGBM          │   │
│   │  ┌─────────┐ │  │   + SHAP + Optuna)  │   + SHAP + Optuna)          │   │
│   │  │ score_  │ │  ├───────────────────────────────────────────────────┤   │
│   │  │ uw_task │ │  │  GraphCollusionDetector (Neo4j / NetworkX)         │   │
│   │  │ score_  │ │  ├───────────────────────────────────────────────────┤   │
│   │  │ cl_task │ │  │  PolicyCopilot (LangGraph)                         │   │
│   │  └─────────┘ │  │   RetrieverAgent → ToolAgent → WriterAgent → HITL  │   │
│   └──────────────┘  └──────────┬───────────────────────────────────────┘   │
│                                 │                                            │
│   ┌─────────────────────────────▼───────────────────────────────────────┐   │
│   │                      Data / Storage Layer                            │   │
│   │  PostgreSQL  │  Qdrant (hybrid BM25+dense)  │  Neo4j  │  Redis      │   │
│   │  HITL JSON   │  Prometheus / Grafana          │  CSV fallback (UI)   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Async Inference Flow

```
Client
  │
  │  POST /underwrite  {features}
  ▼
FastAPI  ──► assert_redis_reachable()  ──►  503 if Redis down (no silent fallback)
  │
  │  apply_async(score_underwriting_task, task_id=<uuid>)
  ▼
Redis Broker
  │
  │  pick up
  ▼
Celery Worker
  ├─ assert_redis_reachable()    ──► RuntimeError (no retry)
  ├─ assert_postgres_reachable() ──► RuntimeError (no retry)
  └─ UnderwritingEngine().predict()  ──► retry ×3 on transient error
  │
  │  store result
  ▼
Redis Backend  ◄──── GET /tasks/{task_id}  ◄──── Client polls
```

---

## 🚀 Quick Start

### Option A — Docker Compose (full stack, recommended)

```bash
# 1. Clone
git clone https://github.com/SumedhPatil1507/claimguard-ai.git
cd claimguard-ai

# 2. Configure
cp .env.example .env
# Edit .env — set GROQ_API_KEY (optional), leave others for local dev

# 3. Launch (Redis, Celery worker, API, Streamlit, Postgres, Neo4j, Prometheus, Grafana)
docker compose up --build

# Services
#   Streamlit UI   → http://localhost:8501
#   FastAPI docs   → http://localhost:8000/docs
#   Grafana        → http://localhost:3000  (admin/admin)
#   Prometheus     → http://localhost:9090
```

### Option B — Local development

```bash
# Python 3.11+
pip install -r requirements.txt

# Generate synthetic data
python data/synthetic_generator.py

# Terminal 1 — Redis (Docker)
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 2 — Celery worker
celery -A src.worker worker --loglevel=info --concurrency=4

# Terminal 3 — FastAPI
uvicorn api.main:app --reload --port 8000

# Terminal 4 — Streamlit
streamlit run app/app.py
```

### Option C — Streamlit Cloud (UI only, no Redis/Celery needed)

The app runs at **https://claimguard-ai-fdiafhzaeme9gr2lndxmv3.streamlit.app**
All ML engines, HITL queue, and compliance dashboard work without any services.
Celery-dependent features show graceful 503 messages.

---

## ⚙️ Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | No | — | Groq `llama3-8b-8192` for Policy Copilot writer; rule-based fallback if unset |
| `REDIS_URL` | Yes (API/worker) | `redis://localhost:6379/0` | Celery broker + result backend |
| `DATABASE_URL` | Yes (production) | — | asyncpg PostgreSQL DSN |
| `SUPABASE_URL` | No | — | Supabase REST endpoint (tier-2 fallback) |
| `SUPABASE_KEY` | No | — | Supabase anon/service key |
| `NEO4J_URI` | No | — | Bolt URI for graph collusion (NetworkX fallback) |
| `NEO4J_USER` | No | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | No | — | Neo4j password |
| `ENCRYPTION_KEY` | No | auto-generated | Fernet key for PII encryption |
| `API_KEYS` | No | demo keys | JSON `{"key":"role"}` map for RBAC |
| `CELERY_TASK_TIMEOUT` | No | `300` | Hard time limit per Celery task (seconds) |
| `CLAIMGUARD_VS_QDRANT_MODE` | No | `local` | `memory` / `local` / `remote` |
| `CLAIMGUARD_VS_RERANKER_BACKEND` | No | `cross_encoder` | `none` / `cross_encoder` / `cohere` |
| `CLAIMGUARD_VS_COHERE_API_KEY` | No | — | Cohere API key for rerank backend |

---

## 📡 API Reference

### Demo API Keys

| Key | Role | Permissions |
|---|---|---|
| `admin-key-demo` | admin | All endpoints |
| `analyst-key-demo` | analyst | All except `/metrics` |
| `viewer-key-demo` | viewer | `/compliance`, `/health` |

### Endpoints

<details>
<summary><b>GET /health</b> — Liveness probe (no auth)</summary>

```bash
curl http://localhost:8000/health
```
```json
{ "status": "ok", "version": "1.0.0", "timestamp": "2026-10-05T10:00:00+00:00" }
```
</details>

<details>
<summary><b>POST /underwrite</b> — Enqueue underwriting job → HTTP 202</summary>

```bash
curl -X POST http://localhost:8000/underwrite \
  -H "X-API-Key: analyst-key-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35, "annual_income": 800000, "credit_score": 720,
    "sum_insured": 1000000, "coverage_type": "motor",
    "num_dependents": 2, "prior_claims_count": 0,
    "region": "north", "occupation": "salaried"
  }'
```
```json
{
  "task_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "PENDING",
  "status_url": "http://localhost:8000/tasks/f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "message": "Underwriting job enqueued. Poll status_url for the result."
}
```
</details>

<details>
<summary><b>POST /claims/score</b> — Enqueue fraud-scoring job → HTTP 202</summary>

```bash
curl -X POST http://localhost:8000/claims/score \
  -H "X-API-Key: analyst-key-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-001", "claimant_id": "CLT-001", "policy_id": "POL-001",
    "claim_amount": 75000, "days_since_policy_start": 45,
    "num_prior_claims": 2, "claim_type": "motor", "claim_severity": "high",
    "repair_shop_id": "SHOP-001"
  }'
```
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "PENDING",
  "status_url": "http://localhost:8000/tasks/a1b2c3d4-..."
}
```
</details>

<details>
<summary><b>GET /tasks/{task_id}</b> — Poll task status</summary>

```bash
curl http://localhost:8000/tasks/f47ac10b-... \
  -H "X-API-Key: analyst-key-demo"
```

| `status` | Meaning |
|---|---|
| `PENDING` | Queued, worker has not started |
| `STARTED` | Worker is running inference |
| `SUCCESS` | Complete — `result` key has payload |
| `FAILURE` | Failed — `error` and `error_type` explain why |
| `RETRY` | Retrying after transient error |

```json
{
  "task_id": "f47ac10b-...",
  "status": "SUCCESS",
  "result": {
    "risk_tier": "low", "risk_score": 0.22,
    "premium_adjustment": 0.95, "shap_drivers": [...],
    "model_version": "xgb-lgbm-v1"
  }
}
```
</details>

<details>
<summary><b>POST /copilot/decide</b> — Run Policy Copilot inline</summary>

```bash
curl -X POST http://localhost:8000/copilot/decide \
  -H "X-API-Key: analyst-key-demo" \
  -H "Content-Type: application/json" \
  -d '{"query": "Assess risk for motor policy applicant", "context_type": "underwriting", "features": {...}}'
```
</details>

<details>
<summary><b>GET /graph/collusion-rings</b> — Detect claim-ring collusion</summary>

```bash
curl http://localhost:8000/graph/collusion-rings \
  -H "X-API-Key: analyst-key-demo"
```
> ⚠️ Requires PostgreSQL — returns 503 if `DATABASE_URL` is unset.
</details>

<details>
<summary><b>GET /compliance</b> — IRDAI compliance report</summary>

```bash
curl http://localhost:8000/compliance -H "X-API-Key: viewer-key-demo"
```
</details>

<details>
<summary><b>GET /metrics</b> — Prometheus metrics (admin only)</summary>

```bash
curl http://localhost:8000/metrics -H "X-API-Key: admin-key-demo"
```
</details>

---

## 🖥️ Streamlit UI Tabs

| Tab | Description | Key Charts |
|---|---|---|
| **📊 Explorer** | EDA on 500 synthetic claims + 300 policies | Fraud distribution histogram, scatter plot, correlation heatmap, risk tier bar |
| **🏦 Underwrite** | Real-time risk scoring form | Risk score gauge (Plotly Indicator), SHAP waterfall horizontal bar |
| **🔍 Claims** | Fraud scoring form | Fraud score gauge, SHAP feature importance bar |
| **🤖 Policy Copilot** | LangGraph agent chat interface | Retrieved policy clauses, model evidence table |
| **🕸️ Graph Intel** | Collusion ring network | Interactive Plotly network graph (claimant nodes + entity diamonds) |
| **👤 HITL Review** | Analyst approve/reject/escalate queue | Queue depth metric, item cards with inline review |
| **📈 Observability** | Live Prometheus metrics | Latency percentile bar, decisions time-series line |
| **✅ Compliance** | IRDAI 10-control dashboard | Score gauge, compliant/partial/non_compliant pie, expandable control cards |

---

## 🧠 ML Layer

### Ensemble Architecture

```
Input Features
     │
     ├──► XGBoost Classifier ──►┐
     │                          ├──► Soft-vote average ──► Risk/Fraud Score
     └──► LightGBM Classifier ──►┘
                    │
                    ▼
               SHAP TreeExplainer
                    │
                    ▼
           Top-5 feature drivers
    [{feature, shap_value, direction}, ...]
```

### Underwriting Features

`age` · `annual_income` · `credit_score` · `sum_insured` · `coverage_type` · `num_dependents` · `prior_claims_count` · `region` · `occupation`

### Claims Features

`claim_amount` · `days_since_policy_start` · `num_prior_claims` · `claim_type` · `claim_severity` · `repair_shop_id` · `medical_provider_id`

### Fallback chain

```
XGBoost+LightGBM → scikit-learn RandomForest → Rule-based heuristic
```

---

## 🔍 Hybrid Vector Search (Qdrant)

```
Query
  │
  ├──► Dense: sentence-transformers/all-MiniLM-L6-v2
  │         └──► Qdrant cosine similarity search  (top-10)
  │
  └──► Sparse: BM25Okapi (rank-bm25)               (top-10)
                        │
                        ▼
              Reciprocal Rank Fusion (RRF, k=60)
                        │
                        ▼
              CrossEncoder re-ranking
              (cross-encoder/ms-marco-MiniLM-L-6-v2)
                        │
                        ▼
               Final top-3 results
    [{content, source, score}, ...]
```

Configure via `CLAIMGUARD_VS_*` env vars — see [Configuration](#️-configuration).

---

## 🕸️ Graph Collusion Detection

The `GraphCollusionDetector` builds a **bipartite graph** linking claimants to shared entities (repair shops, medical providers, witnesses). Suspicious rings are clusters where ≥ 3 claims share ≥ 2 entities.

```
Claimant A ──── SHOP-001 ──── Claimant B
    │                              │
    └──── SHOP-001, DR-042 ────────┘
                                 ▲
                            Colluding ring (severity: high)
```

Metrics: degree centrality, betweenness centrality. Primary: Neo4j. Fallback: NetworkX in-process.

---

## 🔒 Security

| Layer | Implementation |
|---|---|
| Authentication | `X-API-Key` header → role resolution (`admin` / `analyst` / `viewer`) |
| Authorization | `require_roles([...])` FastAPI dependency per endpoint |
| Rate limiting | Sliding-window `RateLimiter` (100 req/60 s per IP) |
| PII encryption | Fernet symmetric encryption (`src/encryption.py`) |
| Production routes | Hard infra checks — 503 if Redis/Postgres unreachable (no CSV fallback) |

---

## 📈 Observability

Prometheus metrics under `claimguard_copilot_*` namespace:

| Metric | Type | Labels |
|---|---|---|
| `claimguard_copilot_agent_runs_total` | Counter | `agent_name` |
| `claimguard_copilot_agent_latency_seconds` | Histogram | `agent_name` |
| `claimguard_copilot_tool_calls_total` | Counter | `tool_name` |
| `claimguard_copilot_retriever_hits_total` | Counter | — |
| `claimguard_copilot_decisions_drafted_total` | Counter | — |
| `claimguard_copilot_hitl_queue_depth` | Gauge | — |
| `claimguard_copilot_graph_queries_total` | Counter | — |

Grafana dashboard JSON available at `http://localhost:3000` after `docker compose up`.

---

## ✅ IRDAI Compliance Module

10 real IRDAI regulatory controls with evidence + remediation:

| Control | Category | Default Status |
|---|---|---|
| IRDAI-001 · Claim settlement within 30 days | Claims Management | ⚠️ Partial |
| IRDAI-002 · KYC verification | Customer Due Diligence | ✅ Compliant |
| IRDAI-003 · Grievance redressal within 15 days | Customer Service | ⚠️ Partial |
| IRDAI-004 · Policy issuance SLA | Operations | ✅ Compliant |
| IRDAI-005 · Premium refund on cancellation | Financial | ✅ Compliant |
| IRDAI-006 · Free-look period enforcement | Customer Rights | ✅ Compliant |
| IRDAI-007 · Portability rights | Customer Rights | ⚠️ Partial |
| IRDAI-008 · Anti-money-laundering checks | Fraud Prevention | ❌ Non-compliant |
| IRDAI-009 · Fraud reporting to IRDAI | Regulatory Reporting | ⚠️ Partial |
| IRDAI-010 · Data localization | Data Governance | ❌ Non-compliant |

---

## 🧪 Testing

```bash
# Run full suite (149 tests, ~20 s)
pytest tests/ -v

# Run specific groups
pytest tests/test_underwriting.py    # ML engine
pytest tests/test_claims_fraud.py    # Fraud engine
pytest tests/test_vector_store.py    # Qdrant hybrid search
pytest tests/test_worker.py          # Celery tasks + API endpoints
pytest tests/test_compliance.py      # IRDAI compliance
pytest tests/test_agent_graph.py     # LangGraph copilot
```

Current status: **149 passed · 6 skipped (sentence-transformers not installed locally) · 0 failed**

---

## 📁 Project Structure

```
claimguard-ai/
├── api/
│   └── main.py              FastAPI app (async Celery enqueue, RBAC, rate-limit)
├── app/
│   └── app.py               Streamlit UI (8 tabs, Plotly dark theme)
├── src/
│   ├── agent_graph.py        LangGraph Policy Copilot pipeline
│   ├── claims_fraud.py       Fraud detection ML engine
│   ├── compliance_irdai.py   IRDAI compliance reporting
│   ├── copilot_metrics.py    Prometheus metrics helpers
│   ├── database.py           DB manager (asyncpg → Supabase → CSV)
│   │                         + ProductionDatabaseManager (no fallback)
│   ├── encryption.py         Fernet PII encryption
│   ├── graph_collusion.py    Neo4j / NetworkX collusion detection
│   ├── hitl.py               Human-in-the-loop review queue
│   ├── rate_limit.py         Sliding-window rate limiter
│   ├── rbac.py               Role-based access control
│   ├── shap_utils.py         SHAP explainability
│   ├── underwriting.py       Underwriting risk-scoring engine
│   ├── vector_store.py       Qdrant hybrid BM25+dense+rerank store
│   ├── vector_store_settings.py  Pydantic settings (CLAIMGUARD_VS_*)
│   └── worker.py             Celery app + background tasks
├── data/
│   ├── synthetic_generator.py  Generate sample_claims.csv + sample_policies.csv
│   ├── policy_docs/            Sample policy wording + IRDAI guidelines
│   └── models/                 Persisted ML model artefacts
├── tests/                    pytest test suite (149 tests)
├── eval/
│   └── ragas_eval.py         RAGAS retrieval quality evaluation
├── docker-compose.yml        Full stack (Redis, Celery, API, Streamlit, PG, Neo4j, Prom, Grafana)
├── Dockerfile
├── requirements.txt          Streamlit Cloud compatible
├── requirements-full.txt     Full local / Docker stack
└── .env.example
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI 0.111 + Pydantic v2 + uvicorn |
| **UI** | Streamlit 1.35 + Plotly 5 (dark theme, interactive) |
| **Async tasks** | Celery 5.4 + Redis 7 |
| **ML** | XGBoost 2.0 + LightGBM 4.3 + SHAP + Optuna |
| **Agent / RAG** | LangGraph 0.1 + LangChain Core + Groq llama3-8b-8192 |
| **Vector store** | Qdrant 1.9 (hybrid: dense + BM25 + CrossEncoder) |
| **Graph** | Neo4j 5 / NetworkX 3.3 |
| **Database** | PostgreSQL 15 via asyncpg |
| **Security** | Fernet encryption + RBAC + rate limiting |
| **Observability** | Prometheus + Grafana |
| **Infra** | Docker Compose + GitHub Actions CI |
| **Tests** | pytest 8 + httpx |

---

## 📄 License

MIT © 2024 ClaimGuard AI — [Sumedh Patil](https://github.com/SumedhPatil1507)

---

<div align="center">

**[⬆ Back to top](#️-claimguard-ai)**

Made with ☕ and 🛡️ for the InsurTech community

</div>
