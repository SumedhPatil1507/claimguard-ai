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
[![Tests](https://img.shields.io/badge/Tests-208%20passing-brightgreen?style=for-the-badge&logo=pytest)](tests/)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?style=for-the-badge&logo=next.js)](frontend/)

<br/>

> **ClaimGuard AI** combines ensemble ML, agentic RAG, and graph analytics to automate and explain the two most critical insurance workflows: **underwriting risk scoring** at policy-issuance time and **claims fraud detection** at filing time — all behind a mandatory Human-in-the-Loop review gate.

<br/>

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [📡 API Reference](#-api-reference) · [🖥️ UI Tabs](#️-streamlit-ui-tabs) · [⚙️ Configuration](#️-configuration) · [🧪 Testing](#-testing)

</div>

---

## ✨ What's New

| Version | Highlights |
|---------|-----------|
| **v1.6** | 🖥️ **Next.js 14 dashboard** — React + Tailwind + Shadcn UI, React Flow graph, JWT auth, async polling UI |
| **v1.5** | 📊 **MLflow + Evidently AI** — experiment tracking (ROC-AUC, F1, artifacts) + data/concept drift detection exposed via Prometheus |
| **v1.4** | 🕸️ **Heterogeneous R-GCN GNN** — two-stage collusion detection; GNN score bar chart, ring colour-coding, severity upgrade rules |
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
| **🕸️ Graph Intel** | Two-stage collusion detection: structural rings + R-GCN GNN re-scoring | GNN score bar chart with thresholds, network graph (node size ∝ GNN risk), severity filter toggle, per-ring mini bar charts |
| **👤 HITL Review** | Analyst approve/reject/escalate queue | Queue depth metric, item cards with inline review |
| **📈 Observability** | Prometheus metrics + Drift Monitoring + MLflow runs | Latency bar, decisions time-series, per-feature drift bar chart (🔴 drifted / 🟢 stable), MLflow run table |
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

ClaimGuard AI uses a **two-stage pipeline** to detect claim-ring collusion:

### Stage 1 — Structural ring detection (always runs)

Builds a bipartite graph linking claimants to shared entities (repair shops, medical providers, witnesses). Suspicious rings are clusters where ≥ 3 claims share ≥ 2 entities.

```
Claimant A ──── SHOP-001 ──── Claimant B
    │                              │
    └──── SHOP-001, DR-042 ────────┘
                                 ▲
                            Ring detected (severity based on size)
```

Primary backend: **Neo4j**. Fallback: **NetworkX** (in-process, no server).

### Stage 2 — R-GCN GNN re-scoring (optional)

A **Relational Graph Convolutional Network** trained on the heterogeneous claims graph upgrades ring severity based on learned collusion signals.

```
Graph schema:

  Node types          Features (dim)
  ──────────          ──────────────────────────────────────────────
  claimant            claim_amount, days_since_policy, prior_claims,
                      claim_type, claim_severity, fraud_label   (6)
  garage              avg_amount, claim_count, fraud_rate,
                      severity_high_rate                        (4)
  medical             same as garage                            (4)

  Edge types (8 total, 4 + 4 reverse)
  ─────────────────────────────────────────────
  claimant ──filed_at_garage──► garage
  claimant ──treated_by──────► medical
  garage   ──co_used_by──────► garage
  medical  ──co_used_by──────► medical
```

```
R-GCN architecture:

  Input projections (per node type → hidden_dim)
       │
  HeteroConv(SAGEConv per relation) → ReLU + Dropout
       │
  HeteroConv(SAGEConv per relation) → ReLU
       │
  claimant embeddings
       │
  Linear(hidden → hidden/2) → ReLU → Dropout → Linear(hidden/2 → 1)
       │
  sigmoid → collusion_score ∈ (0, 1)
```

**Severity upgrade rules** after GNN scoring:
- `max_gnn_score ≥ 0.80` → ring severity forced to **high**
- `max_gnn_score ≥ 0.60` + current severity `low` → upgraded to **medium**

### Training the GNN

```bash
# Quick start (uses data/sample_claims.csv)
python scripts/train_gnn_collusion.py

# Full run with Neo4j export
python scripts/train_gnn_collusion.py \
  --claims-csv data/sample_claims.csv \
  --epochs 100 --hidden-dim 64 --lr 1e-3 \
  --write-neo4j \
  --scores-out data/gnn_scores.json

# Install PyG first (CPU)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
```

| Flag | Default | Description |
|---|---|---|
| `--claims-csv` | `data/sample_claims.csv` | Input data path |
| `--db-url` | — | PostgreSQL DSN (overrides CSV) |
| `--epochs` | 100 | Max training epochs |
| `--hidden-dim` | 64 | GNN hidden dimension |
| `--lr` | 1e-3 | Adam learning rate |
| `--patience` | 15 | Early-stopping patience |
| `--write-neo4j` | off | Write scores to Neo4j |
| `--scores-out` | — | Export JSON score file |
| `--resume` | off | Resume from checkpoint |

**Exit codes**: `0` = success · `1` = data/model error · `2` = PyG absent (soft, not a CI blocker)



---

## 🖥️ Next.js 14 Dashboard (v1.6)

A production-quality React dashboard that runs alongside the Streamlit UI. Both coexist — Streamlit on `:8501`, Next.js on `:3001`.

### Tech stack

| Layer | Technology |
|---|---|
| Framework | Next.js 14 (App Router, `output: standalone`) |
| Styling | Tailwind CSS v3 · dark navy/teal design system |
| Components | Shadcn UI design tokens + custom Radix UI primitives |
| Charts | Recharts — bar, pie, scatter, radial gauge, SHAP waterfall |
| Graph | **React Flow v11** — interactive force-directed canvas with minimap |
| Forms | react-hook-form + Zod validation |
| Auth | JWT (`/auth/token` → `Authorization: Bearer <token>`) |
| Async jobs | Task-ID polling (`/tasks/{id}`) with `useTaskPoller` hook |

### Pages

| Route | Description |
|---|---|
| `/login` | JWT login. Demo: `admin/admin123` · `analyst/analyst123` · `viewer/viewer123` |
| `/explorer` | Claims data table (search + pagination) + 4 Recharts: type breakdown, fraud pie, severity, prior-claims scatter |
| `/underwriting` | Feature form → `POST /underwrite` (202) → Celery poll → risk gauge + SHAP bar |
| `/fraud` | Claim form → `POST /claims/score` (202) → Celery poll → fraud gauge + alert banner + SHAP |
| `/graph` | `GET /graph/collusion-rings` → React Flow canvas — claimant/entity nodes, ring colour groups, animated high-severity edges |
| `/hitl` | `GET /hitl/queue` + `POST /hitl/review/{id}` → analyst approve / reject / escalate cards |

### JWT auth

```bash
# Get a token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"analyst","password":"analyst123"}'
# → {"access_token":"<jwt>","role":"analyst","expires_in":28800}

# Use on all subsequent requests
Authorization: Bearer <access_token>

# Legacy X-API-Key still works (no breaking change)
X-API-Key: analyst-key-demo
```

Configure via env vars:
- `JWT_SECRET_KEY` — signing secret (`openssl rand -hex 32` to generate)
- `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` — default 480 (8 h)

### Quick start (local)

```bash
cd frontend
npm install --legacy-peer-deps
npm run dev          # → http://localhost:3001

# FastAPI must be running for real data:
uvicorn api.main:app --reload --port 8000
```

### Quick start (Docker Compose)

```bash
docker compose up --build
# Next.js dashboard → http://localhost:3001
# Streamlit UI      → http://localhost:8501
# FastAPI + docs    → http://localhost:8000/docs
```

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

## 📊 MLflow Experiment Tracking

Every `_train()` call in `UnderwritingEngine` and `FraudDetectionEngine` logs to MLflow:

| Artifact | Description |
|---|---|
| **Parameters** | `n_estimators`, `max_depth`, `learning_rate`, `n_samples`, `n_features`, `fallback_chain` |
| **Metrics** | `roc_auc_cv` (3-fold cross-val), `f1_cv`, `fraud_pos_rate` (fraud model only) |
| **Model artifact** | Pickled ensemble as `model/<name>.pkl` |
| **Feature importance** | Bar chart PNG at `plots/feature_importance.png` |
| **Tags** | `model_type`, `framework`, `model_version`, `fraud_threshold` |

```bash
# Start MLflow tracking server (optional — defaults to ./mlruns)
mlflow server --host 0.0.0.0 --port 5000

# Point the engines at it
export MLFLOW_TRACKING_URI=http://localhost:5000

# Trigger a training run
python -c "from src.underwriting import UnderwritingEngine; UnderwritingEngine(force_retrain=True)"

# Open the UI
open http://localhost:5000
```

Experiments: `claimguard_underwriting` · `claimguard_fraud_detection`

---

## 🌊 Data & Concept Drift Monitoring

`src/drift_monitor.py` implements two drift detection strategies:

| Mode | Library | Numeric | Categorical | Concept Drift |
|---|---|---|---|---|
| Primary | Evidently AI | DataDriftPreset | DataDriftPreset | TargetDriftPreset |
| Fallback | scipy / numpy | KS test | Chi-squared | PSI |

```python
from src.drift_monitor import DriftMonitor, save_baseline

# 1. Save training distribution as baseline (called automatically in _train())
save_baseline(train_df, model_name="fraud", prediction_col="fraud_label")

# 2. Detect drift on incoming batch
monitor = DriftMonitor("fraud")
report  = monitor.detect(incoming_df, predictions=fraud_scores)

print(report.data_drift_score)       # 0.0 – 1.0 share of drifted features
print(report.drifted_features)       # ['claim_amount', ...]
print(report.concept_drift_detected) # True / False
print(report.concept_drift_score)    # PSI of prediction distribution
```

**Prometheus gauges** (updated after every `detect()` call):

| Metric | Labels | Description |
|---|---|---|
| `claimguard_drift_data_score` | `model` | Share of drifted features |
| `claimguard_drift_feature_score` | `model`, `feature` | Per-feature drift score |
| `claimguard_drift_concept_score` | `model` | PSI of predictions |
| `claimguard_drift_concept_detected` | `model` | 1.0 = drift detected |

Configure thresholds via env vars:
```bash
CLAIMGUARD_DRIFT_DATA_THRESHOLD=0.10     # feature-level KS/chi2 threshold
CLAIMGUARD_DRIFT_CONCEPT_THRESHOLD=0.25  # PSI threshold for concept drift
```

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

Current status: **208 passed · 24 skipped (torch DLL broken locally / sentence-transformers not installed) · 0 failed**

---

## 📁 Project Structure

```
claimguard-ai/
├── api/
│   └── main.py              FastAPI app (async Celery enqueue, RBAC, rate-limit)
├── app/
│   └── app.py               Streamlit UI (8 tabs, Plotly dark theme, interactive charts)
├── src/
│   ├── agent_graph.py        LangGraph Policy Copilot pipeline
│   ├── claims_fraud.py       Fraud detection ML engine
│   ├── compliance_irdai.py   IRDAI compliance reporting
│   ├── copilot_metrics.py    Prometheus metrics helpers
│   ├── database.py           DB manager + ProductionDatabaseManager (no fallback)
│   ├── drift_monitor.py      Evidently AI + scipy drift detection, Prometheus export (NEW)
│   ├── encryption.py         Fernet PII encryption
│   ├── gnn_collusion.py      Heterogeneous R-GCN GNN for collusion scoring (NEW)
│   ├── graph_collusion.py    Two-stage ring detector (structural + GNN)
│   ├── hitl.py               Human-in-the-loop review queue
│   ├── rate_limit.py         Sliding-window rate limiter
│   ├── rbac.py               Role-based access control
│   ├── shap_utils.py         SHAP explainability
│   ├── underwriting.py       Underwriting risk-scoring engine
│   ├── vector_store.py       Qdrant hybrid BM25+dense+rerank store
│   ├── vector_store_settings.py  Pydantic settings (CLAIMGUARD_VS_*)
│   └── worker.py             Celery app + background tasks
├── scripts/
│   └── train_gnn_collusion.py  R-GCN training pipeline (CLI)
├── data/
│   ├── synthetic_generator.py  Generate sample_claims.csv + sample_policies.csv
│   ├── policy_docs/            Sample policy wording + IRDAI guidelines
│   └── models/                 Persisted ML model artefacts
├── tests/                    pytest test suite (208 tests)
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
| **Experiment tracking** | MLflow 2.14 (params, metrics, artifacts, feature importance) |
| **Drift monitoring** | Evidently AI 0.4 + scipy KS/chi² + PSI fallback |
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
