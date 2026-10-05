# 🛡️ ClaimGuard AI

> Agentic InsurTech Platform for Underwriting Risk Scoring & Claims Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1.5-green)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

ClaimGuard AI is an end-to-end agentic InsurTech platform that combines ensemble machine learning (XGBoost + LightGBM) with a LangGraph-powered Policy Copilot to automate and explain two critical insurance workflows: underwriting risk scoring at policy-issuance time and fraud scoring at claim-filing time. The platform indexes policy wording and IRDAI regulatory text into a ChromaDB vector store for retrieval-augmented generation, runs graph-based collusion ring detection over shared claimant entities via Neo4j or NetworkX, enforces a human-in-the-loop (HITL) review queue so no AI decision is ever auto-approved or auto-denied, and exposes everything through a FastAPI backend with RBAC and a Streamlit UI with eight interactive tabs — all observable via Prometheus and Grafana.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ClaimGuard AI                               │
│                                                                      │
│  Policy Docs / IRDAI Text                                           │
│         │                                                            │
│         ▼                                                            │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐   │
│  │  ChromaDB   │    │          LangGraph Pipeline               │   │
│  │  VectorStore│◄───│  RetrieverAgent → ToolAgent → WriterAgent │   │
│  └─────────────┘    │          → HITLRouter                    │   │
│                      └──────────────────────────────────────────┘   │
│  Claims CSV / Policies CSV                                          │
│         │                                                            │
│         ▼                                                            │
│  ┌────────────────┐   ┌──────────────────┐   ┌───────────────┐     │
│  │ Underwriting   │   │  Fraud Detection │   │ Graph Collusion│    │
│  │ Engine (XGB+   │   │  Engine (XGB+    │   │ Detector       │    │
│  │  LGBM+SHAP)   │   │  LGBM+SHAP)      │   │ (Neo4j/NX)    │    │
│  └────────────────┘   └──────────────────┘   └───────────────┘     │
│         │                      │                      │              │
│         └──────────────────────┴──────────────────────┘              │
│                               │                                      │
│                        ┌──────▼──────┐                              │
│                        │  FastAPI    │◄── X-API-Key RBAC            │
│                        └──────┬──────┘                              │
│                               │                                      │
│                      ┌────────▼────────┐                           │
│                      │  Streamlit UI   │  8 interactive tabs        │
│                      └─────────────────┘                           │
│                                                                      │
│  Prometheus/Grafana                 HITL Queue (PostgreSQL/JSON)   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Docker Compose (Recommended)

```bash
git clone https://github.com/SumedhPatil1507/claimguard-ai.git
cd claimguard-ai
cp .env.example .env  # Edit with your API keys
docker compose up --build
# API:     http://localhost:8000
# UI:      http://localhost:8501
# Grafana: http://localhost:3000
```

### Local Development

```bash
git clone https://github.com/SumedhPatil1507/claimguard-ai.git
cd claimguard-ai
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python data/synthetic_generator.py   # Generate synthetic training data
uvicorn api.main:app --port 8000 &   # Start API server
streamlit run app/app.py             # Start UI (opens browser automatically)
```

> **Note:** All ML models and RAG components degrade gracefully — the app runs without any API keys set. See the fallback behavior table in the Environment Variables section below.

---

## Environment Variables

| Name | Description | Required | Default / Fallback |
|---|---|---|---|
| `GROQ_API_KEY` | Groq LLM API key (llama3-8b-8192) | No | Rule-based narrative template |
| `DATABASE_URL` | PostgreSQL async connection string | No | CSV file fallback |
| `SUPABASE_URL` | Supabase project URL | No | CSV file fallback |
| `SUPABASE_KEY` | Supabase anon/service key | No | CSV file fallback |
| `NEO4J_URI` | Neo4j bolt connection URI | No | NetworkX in-memory fallback |
| `NEO4J_USER` | Neo4j username | No | NetworkX in-memory fallback |
| `NEO4J_PASSWORD` | Neo4j password | No | NetworkX in-memory fallback |
| `ENCRYPTION_KEY` | Fernet symmetric encryption key | No | Auto-generated on startup |
| `API_KEYS` | JSON dict mapping API key → role | No | Demo keys (see API Reference) |

---

## API Reference

All endpoints except `/health` and `/metrics` require an `X-API-Key` header.

| Method | Path | Required Role | Description |
|---|---|---|---|
| `GET` | `/health` | None | Liveness/readiness check |
| `POST` | `/underwrite` | analyst, admin | Score underwriting risk for a new policy applicant |
| `POST` | `/claims/score` | analyst, admin | Score a filed claim for fraud probability |
| `POST` | `/copilot/decide` | analyst, admin | Run the full LangGraph Policy Copilot pipeline |
| `GET` | `/graph/collusion-rings` | analyst, admin | Detect collusion rings in the claims graph |
| `GET` | `/compliance` | viewer, analyst, admin | Generate the IRDAI compliance JSON report |
| `GET` | `/metrics` | admin | Prometheus metrics scrape endpoint |

### Demo API Keys

| Key | Role |
|---|---|
| `admin-key-demo` | admin |
| `analyst-key-demo` | analyst |
| `viewer-key-demo` | viewer |

### Example: Score a Claim

```bash
curl -X POST http://localhost:8000/claims/score \
  -H "X-API-Key: analyst-key-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "claim_id": "CLM-2024-001",
    "claim_amount": 85000,
    "days_since_policy_start": 45,
    "prior_claims_count": 2,
    "claimant_age": 34,
    "policy_type": "health",
    "garage_id": "GRG-077"
  }'
```

### Example: Run Policy Copilot

```bash
curl -X POST http://localhost:8000/copilot/decide \
  -H "X-API-Key: analyst-key-demo" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Is this health claim eligible under the cashless hospitalization clause?",
    "claim_id": "CLM-2024-001",
    "context": {}
  }'
```

---

## Streamlit UI Tabs

| Tab | Description |
|---|---|
| **Explorer** | Interactive EDA on claims and policy datasets — histograms, correlation heatmaps, and distribution plots. |
| **Underwrite** | Enter applicant/policy features and receive an instant risk tier (Low / Medium / High) with premium adjustment recommendation and SHAP waterfall chart. |
| **Claims** | Submit a claim and receive a fraud probability score with SHAP feature importance and a confidence band. |
| **Policy Copilot** | Chat interface for the LangGraph agent — ask natural language questions about policy coverage and receive structured decisions citing retrieved clauses and model evidence. |
| **Graph Intel** | Interactive NetworkX / Neo4j collusion ring visualization — nodes are claimants, garages, and medical providers; edges encode shared-claim relationships. |
| **HITL** | Human-in-the-loop analyst review queue — approve, reject, or escalate AI-drafted decisions before any action is taken. |
| **Observability** | Live Prometheus metrics displayed as Plotly charts — agent latency percentiles, tool call counts, retriever hit rates, and HITL queue depth. |
| **Compliance** | IRDAI regulatory dashboard with a compliance score gauge and per-control status cards showing evidence and remediation guidance. |

---

## Policy Copilot — LangGraph Pipeline

The Policy Copilot is a four-node LangGraph pipeline that runs for every underwriting or claims decision:

```
RetrieverAgent
    │  Semantic search over ChromaDB (policy docs + IRDAI text)
    ▼
ToolAgent
    │  Calls underwriting / fraud-scoring models + SHAP explainer
    ▼
WriterAgent
    │  Synthesizes a structured decision citing retrieved clause + model evidence
    ▼
HITLRouter
    └─ Enqueues draft decision for human analyst review
```

**Graceful degradation:**
- LangGraph not installed → identical linear pipeline without the graph runtime
- ChromaDB / sentence-transformers not installed → retriever returns empty list, pipeline continues
- `GROQ_API_KEY` not set → WriterAgent uses a rule-based narrative template
- Model or SHAP unavailable → those fields are null, nothing raises an exception

---

## ML Models

### Underwriting Engine (`src/underwriting.py`)

- **Input:** Applicant demographics, policy type, coverage amount, claims history, credit proxy features
- **Model:** XGBoost + LightGBM ensemble (soft-vote averaging), tuned with Optuna
- **Output:** Risk tier (Low / Medium / High / Very High), risk score 0–1, premium adjustment factor, SHAP top drivers
- **Training data:** Synthetic policy dataset generated by `data/synthetic_generator.py`

### Claims Fraud Detector (`src/claims_fraud.py`)

- **Input:** Claim amount, days since policy start, prior claims count, claimant age, garage/provider ID, policy type
- **Model:** XGBoost + LightGBM ensemble, tuned with Optuna
- **Output:** Fraud probability 0–1, fraud tier, SHAP explanation, recommended action (auto-pass / review / escalate)
- **Training data:** Synthetic claims dataset with injected fraud patterns

### Graph Collusion Detector (`src/graph_collusion.py`)

Analyzes shared entities (claimants, repair garages, medical providers, witnesses) across claims using degree centrality, betweenness centrality, and connected-component analysis. A garage or provider appearing across many high-severity claims within a short window is flagged as a potential collusion hub.

### SHAP Explainability (`src/shap_utils.py`)

Provides TreeExplainer-based SHAP values for both models, exposed as waterfall charts in the UI and included in every API response and copilot decision narrative.

---

## Compliance — IRDAI Controls

`src/compliance_irdai.py` tracks **10 IRDAI regulatory controls** including:

1. Claim settlement within 30-day turnaround (IRDAI Circular 2015/Claims)
2. Mandatory cashless facility for network hospitals
3. Policy wordings in plain vernacular language
4. Grievance redressal mechanism (IGMS compliance)
5. Anti-fraud policy and fraud monitoring unit
6. KYC norms for policyholder onboarding
7. Data localization and privacy standards
8. Solvency margin maintenance reporting
9. Renewal notice timelines
10. HITL review gate — no automated claim denial without human sign-off

Each control exposes `status`, `evidence`, `remediation`, and `last_checked` fields. The Compliance tab renders a score gauge (0–100) and per-control cards.

---

## Evaluation

Run the RAGAS evaluation to score retrieval precision/recall and answer faithfulness of the Policy Copilot against source policy and IRDAI documents:

```bash
python eval/ragas_eval.py
```

Outputs a JSON report to `eval/results/ragas_report_<timestamp>.json` with:
- **Context Precision** — fraction of retrieved chunks that are relevant
- **Context Recall** — fraction of relevant chunks that were retrieved
- **Answer Faithfulness** — degree to which the WriterAgent's decision is grounded in retrieved text
- **Answer Relevancy** — how directly the answer addresses the query

---

## Project Structure

```
claimguard-ai/
├── src/
│   ├── underwriting.py        # Risk scoring engine (XGB + LGBM + SHAP)
│   ├── claims_fraud.py        # Claims fraud scoring engine
│   ├── graph_collusion.py     # Collusion ring detector (Neo4j / NetworkX)
│   ├── shap_utils.py          # SHAP explainability utilities
│   ├── vector_store.py        # ChromaDB vector store + embedding
│   ├── agent_graph.py         # LangGraph Policy Copilot pipeline
│   ├── copilot_metrics.py     # Prometheus metrics (claimguard_copilot_*)
│   ├── compliance_irdai.py    # IRDAI compliance module
│   ├── hitl.py                # HITL review queue logic
│   ├── database.py            # asyncpg / Supabase / CSV fallback
│   ├── encryption.py          # Fernet encryption helpers
│   ├── rbac.py                # Role-based access control
│   └── rate_limit.py          # Rate limiting middleware
├── api/
│   └── main.py                # FastAPI application
├── app/
│   └── app.py                 # Streamlit UI (8 tabs)
├── data/
│   └── synthetic_generator.py # Synthetic claims + policy dataset generator
├── eval/
│   └── ragas_eval.py          # RAGAS evaluation script
├── tests/
│   ├── test_underwriting.py
│   ├── test_claims_fraud.py
│   ├── test_graph_collusion.py
│   ├── test_agent_graph.py
│   └── test_api.py
├── grafana/
│   └── dashboards/            # Grafana dashboard JSON exports
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI pipeline
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML | XGBoost, LightGBM, Scikit-learn, SHAP, Optuna |
| Agent | LangGraph, LangChain Core, Groq llama3-8b-8192 |
| RAG | ChromaDB, sentence-transformers (all-MiniLM-L6-v2) |
| API | FastAPI, Pydantic v2, uvicorn |
| UI | Streamlit, Plotly |
| Database | PostgreSQL (asyncpg), Supabase, CSV fallback |
| Graph | Neo4j, NetworkX |
| Observability | Prometheus, Grafana |
| Security | Fernet encryption, RBAC, rate limiting |
| Infrastructure | Docker, Docker Compose, GitHub Actions |

---

## CI / CD

GitHub Actions runs on every push and pull request to `main`:

```yaml
# .github/workflows/ci.yml
- Lint with ruff
- Type-check with mypy
- Run pytest (tests/)
- Build Docker image
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Commit with conventional messages: `feat:`, `fix:`, `docs:`, `refactor:`
4. Open a pull request against `main`

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*ClaimGuard AI is a research and demonstration platform. All decisions produced by the AI pipeline are enqueued for mandatory human review and must not be used as the sole basis for any insurance underwriting or claims adjudication decision.*
