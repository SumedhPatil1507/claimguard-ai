<div align="center">

# 🛡️ ClaimGuard AI

### Agentic InsurTech Platform · Underwriting Risk Scoring · Claims Fraud Detection · Real-Time SSE Streaming

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14%20App%20Router-black?style=for-the-badge&logo=next.js)](frontend/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](streamlit_app.py)
[![Celery](https://img.shields.io/badge/Celery-5.4-37814A?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![Redis](https://img.shields.io/badge/Redis-7%20Pub%2FSub-DC382D?style=for-the-badge&logo=redis&logoColor=white)](https://redis.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge&logo=xgboost)](https://xgboost.ai)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1-00A67E?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.9-DB4437?style=for-the-badge&logo=qdrant)](https://qdrant.tech)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-268%20passing-brightgreen?style=for-the-badge&logo=pytest)](tests/)

<br/>

> **ClaimGuard AI** combines ensemble machine learning (XGBoost, LightGBM, TreeSHAP), agentic RAG (LangGraph, Qdrant hybrid search), heterogeneous graph neural networks (R-GCN, Neo4j), and real-time **Server-Sent Events (SSE)** via **Redis Pub/Sub** to automate and explain insurance workflows: **underwriting risk scoring** at policy-issuance time and **claims fraud detection** at filing time — protected by a mandatory Human-in-the-Loop review gate.

<br/>

[🚀 Quick Start](#-quick-start) · [🏗️ Architecture](#️-architecture) · [📡 SSE Streaming & API Reference](#-real-time-sse-streaming--api-reference) · [🖥️ Next.js 14 Frontend](#️-nextjs-14-production-frontend) · [🎈 Streamlit Interactive Platform](#-interactive-streamlit-platform) · [⚙️ Configuration](#️-configuration) · [🧪 Testing](#-testing)

</div>

---

## ✨ Highlights & Releases

| Version | Highlights |
|---------|-----------|
| **v1.7** | ⚡ **Real-Time SSE Streaming** — Server-Sent Events with Redis Pub/Sub (`GET /underwrite/stream/{task_id}`, `GET /claims/stream/{task_id}`); custom `useTaskStream` hook with animated live progress indicators. Dual-interface support (Next.js 14 primary + Streamlit runner). |
| **v1.6** | 🖥️ **Next.js 14 Production Dashboard** — Next.js 14 App Router, Tailwind CSS, Shadcn UI, React Flow collusion canvas, JWT auth, and interactive analytics. |
| **v1.5** | 📊 **MLflow + Evidently AI** — Experiment tracking (ROC-AUC, F1, artifacts) + data/concept drift detection exposed via Prometheus. |
| **v1.4** | 🕸️ **Heterogeneous R-GCN GNN** — Two-stage collusion detection; GNN score attribution, ring severity upgrades, and graph centrality. |
| **v1.3** | 🔄 **Celery + Redis Async Queue** — Non-blocking `/underwrite` & `/claims/score` endpoints with instant `task_id` dispatch and Redis result storage. |
| **v1.2** | 🔍 **Hybrid Vector Search** — Qdrant + BM25 sparse search + Cross-Encoder re-ranking (`cross-encoder/ms-marco-MiniLM-L-6-v2`). |
| **v1.0** | 🛡️ **Foundational InsurTech Platform** — ML ensemble, LangGraph Policy Copilot, HITL review queue, and IRDAI compliance engine. |

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph UI["Frontend Presentation Layer"]
        A1["Next.js 14 App Router\n(:3000 / :3001)\nPrimary Production Dashboard"]
        A2["Streamlit Interactive UI\n(:8501)\nstreamlit run streamlit_app.py"]
    end

    subgraph API["Backend & Ingestion Layer (FastAPI :8000)"]
        B1["JWT Auth & Role-Based Access Control"]
        B2["POST /underwrite  (202 Async Enqueue)"]
        B3["POST /claims/score (202 Async Enqueue)"]
        B4["GET /underwrite/stream/{task_id} (SSE)"]
        B5["GET /claims/stream/{task_id} (SSE)"]
        B6["GET /tasks/{task_id} (Status Polling)"]
        B7["POST /copilot/decide (LangGraph Inline)"]
    end

    subgraph ASYNC["Async Worker & Pub/Sub (Redis 7 + Celery)"]
        C1["Redis Broker & Result Store (:6379)"]
        C2["Celery Concurrency Workers (4x)"]
        C3["Redis Pub/Sub Event Dispatcher\n(claimguard:task_events:{task_id})"]
    end

    subgraph DOMAIN["Intelligence & Machine Learning Engines"]
        D1["Underwriting Risk Engine\n(XGBoost + LightGBM + TreeSHAP)"]
        D2["Claims Fraud Detection Engine\n(XGBoost + LightGBM + Optuna)"]
        D3["Graph Collusion Detector\n(Neo4j / NetworkX + Heterogeneous R-GCN)"]
        D4["Policy Copilot Agentic RAG\n(LangGraph + Qdrant + Cross-Encoder)"]
        D5["IRDAI Compliance & Drift Monitor\n(Evidently AI + MLflow + Prometheus)"]
    end

    subgraph STORAGE["Data & Observability Tier"]
        E1[("PostgreSQL 15")]
        E2[("Qdrant Vector DB")]
        E3[("Neo4j Graph DB")]
        E4["Prometheus (:9090) & Grafana (:3002)"]
    end

    A1 -->|REST / SSE with Bearer JWT| B1
    A2 -->|Direct Engine Execution| DOMAIN
    B1 --> B2 & B3 & B4 & B5 & B6 & B7
    B2 & B3 -->|Enqueue Task| C1
    C1 --> C2
    C2 --> D1 & D2
    C2 -->|Publish STARTED, PROGRESS, SUCCESS, FAILURE| C3
    C3 -->|Push Real-Time Events via SSE| B4 & B5
    B7 --> D4
    D3 --> E1 & E3
    D4 --> E2
    D5 --> E4
```

---

## 🚀 Quick Start

### Option 1 — Docker Compose (Full Stack, Production Recommended)

Launch the complete microservices stack (FastAPI backend, Next.js 14 frontend, Celery workers, Redis, PostgreSQL, Neo4j, Prometheus, Grafana) with a single command:

```bash
# 1. Clone repository
git clone https://github.com/SumedhPatil1507/claimguard-ai.git
cd claimguard-ai

# 2. Configure environment
cp .env.example .env

# 3. Build and launch all containers
docker compose up --build
```

#### Services & Ports:
| Service | URL | Description |
|---|---|---|
| **Next.js 14 Frontend** | [`http://localhost:3000`](http://localhost:3000) (or `:3001`) | Primary InsurTech production web dashboard |
| **FastAPI Backend & Docs** | [`http://localhost:8000/docs`](http://localhost:8000/docs) | OpenAPI interactive Swagger UI |
| **Grafana Dashboard** | [`http://localhost:3002`](http://localhost:3002) | Real-time monitoring (`admin` / `admin`) |
| **Prometheus** | [`http://localhost:9090`](http://localhost:9090) | Metrics scraper & alerting |
| **Neo4j Browser** | [`http://localhost:7474`](http://localhost:7474) | Graph visualization (`neo4j` / `neo4jpass`) |

---

### Option 2 — Local Development (Next.js 14 + FastAPI + Celery)

```bash
# Terminal 1 — Dependencies, Data & Redis
pip install -r requirements.txt
python data/synthetic_generator.py
docker run -d -p 6379:6379 redis:7-alpine

# Terminal 2 — Celery Task Queue Worker
celery -A src.worker worker --loglevel=info --concurrency=4

# Terminal 3 — FastAPI Backend (with SSE & Redis Pub/Sub)
uvicorn api.main:app --reload --port 8000

# Terminal 4 — Next.js 14 Frontend
cd frontend
npm install
npm run dev
# Opens at http://localhost:3001
```

---

### Option 3 — Running the Interactive Streamlit Platform

ClaimGuard AI includes a full-featured Streamlit platform with rich interactive Plotly gauges, tabular analyzers, and graph visualizers:

```bash
# Install dependencies
pip install -r requirements.txt
pip install -r legacy/streamlit/requirements.txt

# Run directly from root:
streamlit run streamlit_app.py
# Or run legacy file directly:
streamlit run legacy/streamlit/app.py
```

---

## 📡 Real-Time SSE Streaming & API Reference

### Real-Time Task Streaming via Server-Sent Events (SSE)

ClaimGuard AI streams live task lifecycle events via Redis Pub/Sub directly to clients:

```
Celery Worker                   Redis Pub/Sub                     FastAPI SSE                     Next.js 14
┌────────────┐               ┌─────────────────┐               ┌───────────────┐               ┌────────────┐
│ Task Start │ ──publish───► │ claimguard:     │ ──receive────►│ EventSource   │ ──push stream►│ useTask-   │
│ (25%)      │               │ task_events:    │               │ Response      │               │ Stream()   │
│            │               │ {task_id}       │               │               │               │            │
│ Inference  │ ──publish───► │                 │ ──receive────►│ (text/event-  │ ──push stream►│ Progress:  │
│ (70%)      │               │                 │               │  stream)      │               │ 70% bar    │
│            │               │                 │               │               │               │            │
│ Completed  │ ──publish───► │                 │ ──receive────►│ Close Stream  │ ──push stream►│ Render ML  │
│ (100%)     │               │                 │               │               │               │ Results    │
└────────────┘               └─────────────────┘               └───────────────┘               └────────────┘
```

#### Streaming Endpoints:
- `GET /underwrite/stream/{task_id}?token=<jwt>` — Real-time SSE stream for underwriting risk scoring.
- `GET /claims/stream/{task_id}?token=<jwt>` — Real-time SSE stream for fraud detection.
- `GET /tasks/stream/{task_id}?token=<jwt>` — Generic SSE stream for any enqueued Celery task.

#### SSE Event Payload Example:
```json
{
  "task_id": "c89b70b4-4b55-4674-8fa7-86c262adfc29",
  "status": "PROGRESS",
  "progress": {
    "stage": "Evaluating underwriting risk features & SHAP attribution",
    "percent": 70
  }
}
```

```json
{
  "task_id": "c89b70b4-4b55-4674-8fa7-86c262adfc29",
  "status": "SUCCESS",
  "progress": {
    "stage": "Completed",
    "percent": 100
  },
  "result": {
    "risk_score": 0.245,
    "risk_tier": "low",
    "premium_adjustment": 0.95,
    "shap_drivers": [
      { "feature": "credit_score", "shap_value": -0.18, "direction": "decreases risk" },
      { "feature": "annual_income", "shap_value": -0.12, "direction": "decreases risk" }
    ],
    "model_version": "1.0.0"
  }
}
```

---

### Core REST Endpoints

<details>
<summary><b>1. POST /auth/token</b> — Obtain JWT Access Token</summary>

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username":"analyst","password":"analyst123"}'
```
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "role": "analyst",
  "expires_in": 28800
}
```
</details>

<details>
<summary><b>2. POST /underwrite</b> — Enqueue Underwriting Job (HTTP 202)</summary>

```bash
curl -X POST http://localhost:8000/underwrite \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "annual_income": 800000,
    "credit_score": 720,
    "sum_insured": 1000000,
    "coverage_type": "motor",
    "num_dependents": 2,
    "prior_claims_count": 0,
    "region": "north",
    "occupation": "salaried"
  }'
```
```json
{
  "task_id": "c89b70b4-4b55-4674-8fa7-86c262adfc29",
  "status": "PENDING",
  "status_url": "http://localhost:8000/tasks/c89b70b4-4b55-4674-8fa7-86c262adfc29",
  "message": "Underwriting job enqueued."
}
```
</details>

<details>
<summary><b>3. POST /claims/score</b> — Enqueue Claims Fraud Scoring (HTTP 202)</summary>

```bash
curl -X POST http://localhost:8000/claims/score \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-001",
    "claimant_id": "CLT-001",
    "policy_id": "POL-001",
    "claim_amount": 75000,
    "days_since_policy_start": 45,
    "num_prior_claims": 2,
    "claim_type": "motor",
    "claim_severity": "high",
    "repair_shop_id": "SHOP-001"
  }'
```
</details>

<details>
<summary><b>4. GET /graph/collusion-rings</b> — Collusion Ring Detection</summary>

```bash
curl http://localhost:8000/graph/collusion-rings \
  -H "Authorization: Bearer <token>"
```
</details>

<details>
<summary><b>5. GET /hitl/queue & POST /hitl/review/{item_id}</b> — Human-in-the-Loop Review</summary>

```bash
curl http://localhost:8000/hitl/queue -H "Authorization: Bearer <token>"
```
</details>

---

## 🖥️ Next.js 14 Production Frontend

The frontend is located in [`frontend/`](frontend/) and built on **Next.js 14 App Router**:

```
frontend/src/
├── app/
│   ├── (dashboard)/
│   │   ├── layout.tsx         # Sidebar, live API connectivity monitor, JWT auth state
│   │   ├── explorer/page.tsx  # Interactive claims explorer, filtering & charts
│   │   ├── underwriting/page.tsx # Real-time SSE risk scoring & TreeSHAP breakdown
│   │   ├── fraud/page.tsx     # Real-time SSE fraud probability prediction
│   │   ├── claims/page.tsx    # Dedicated claims route alias
│   │   ├── graph/page.tsx     # React Flow collusion ring network graph
│   │   └── hitl/page.tsx      # IRDAI Human-in-the-Loop review queue
│   ├── login/page.tsx         # JWT credentials login with demo auto-fill
│   └── layout.tsx             # Root layout with AuthProvider context
├── components/
│   ├── charts/                # Recharts gauges, SHAP waterfall bars, scatter plots
│   ├── graph/                 # React Flow canvas, node/edge custom layouts
│   └── ui/                    # Badges, stat cards, dialogs, forms
├── hooks/
│   ├── useAuth.tsx            # Context provider for JWT auth & RBAC
│   ├── useTaskStream.ts       # Native EventSource SSE streaming hook
│   └── useTaskPoller.ts       # Fallback polling hook
└── lib/
    └── api.ts                 # Typed API client & exception wrappers
```

---

## 🎈 Interactive Streamlit Platform

To run the standalone Streamlit analytics platform:

```bash
streamlit run streamlit_app.py
```

### Streamlit Features:
1. **Underwriting Intelligence Tab**: Real-time interactive parameter tweaking with dynamic risk dials and premium adjustment calculations.
2. **Claims Fraud Scanner Tab**: Instant fraud scoring with probability gauges and SHAP waterfall contributions.
3. **Policy Copilot (LangGraph RAG)**: Interactive natural language Q&A against policy wording documents using hybrid retrieval.
4. **Collusion Network Tab**: Interactive network graphs highlighting shared garages, medical clinics, and claimant clusters.
5. **HITL Review Queue Tab**: Analyst dashboard to approve, reject, or escalate decisions with audit notes.
6. **Data Drift & Observability Tab**: Drift distribution curves and KS-test statistics.
7. **IRDAI Compliance Tab**: Real-time compliance scorecards mapping to regulatory guidelines.

---

## ⚙️ Configuration

| Variable | Required | Default | Description |
|---|---|---|---|
| `JWT_SECRET_KEY` | Yes (Production) | Auto-generated | Secret key for JWT signing (`openssl rand -hex 32`) |
| `REDIS_URL` | Yes | `redis://localhost:6379/0` | Redis broker and Pub/Sub connection string |
| `DATABASE_URL` | Yes (Production) | `postgresql://claimguard:claimguard_pass@localhost:5432/claimguard` | PostgreSQL asyncpg DSN |
| `NEO4J_URI` | No | `bolt://localhost:7687` | Neo4j Bolt URI for graph collusion queries |
| `NEO4J_USER` | No | `neo4j` | Neo4j database username |
| `NEO4J_PASSWORD` | No | `neo4jpass` | Neo4j database password |
| `GROQ_API_KEY` | No | — | Optional Groq API key for LLaMA 3 Policy Copilot generation |
| `CLAIMGUARD_VS_QDRANT_MODE` | No | `local` | Qdrant vector store mode (`memory`, `local`, `remote`) |

---

## 🧪 Testing

The repository features comprehensive unit, integration, and streaming test suites:

```bash
# Run complete test suite (268 tests)
pytest -v

# Run SSE streaming test suite
pytest tests/test_sse_stream.py -v

# Run worker and Celery test suite
pytest tests/test_worker.py -v

# Run frontend type-check and production build
cd frontend
npm run type-check
npm run build
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
