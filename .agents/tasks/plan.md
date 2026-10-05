# ClaimGuard AI — Implementation Plan

This plan builds the ClaimGuard AI agentic InsurTech platform from a scaffolded-but-empty repo.
The scaffold step already created all directories and `__init__.py` files.
The work is split into five sequential FEATs, each leaving the codebase in a buildable, testable state.

---

## FEAT-001 — Data Layer (synthetic data + policy docs + requirements)

- [ ] 1. Write `requirements.txt` with all pinned dependencies as specified.
      Files: `requirements.txt`
      Verify: `python -m pip install -r requirements.txt --dry-run` exits 0 (or inspecting the file is sufficient at this step; full install confirmed at CI).

- [ ] 2. Write `data/synthetic_generator.py` that generates `data/sample_claims.csv` (500 rows) and `data/sample_policies.csv` (300 rows) with all specified columns. Use `numpy.random.seed(42)` for reproducibility. Run the generator at end of script.
      Files: `data/synthetic_generator.py`
      Verify: `python data/synthetic_generator.py` → both CSVs appear with correct columns and row counts.

- [ ] 3. Write `data/policy_docs/sample_policy_wording.txt` (~500 words, fictional "ClaimGuard Standard Policy" motor/health wording: coverage clauses, exclusions, claim procedures, waiting periods).
      Write `data/policy_docs/irdai_guidelines.txt` (~400 words summarising IRDAI controls: 30-day claim settlement, grievance redressal, KYC, portability, renewal — labelled demo/training, not legal advice).
      Files: `data/policy_docs/sample_policy_wording.txt`, `data/policy_docs/irdai_guidelines.txt`
      Verify: both files exist and are >200 bytes.

---

## FEAT-002 — ML Layer (underwriting, claims fraud, SHAP, graph collusion)

- [ ] 4. Write `src/shap_utils.py`.
      `explain_prediction(model, features_df, feature_names) -> list[dict]`; top-5 SHAP drivers with `feature`, `shap_value`, `direction`. Graceful fallback returns `[]` if `shap` not installed.
      Files: `src/shap_utils.py`
      Verify: `pytest tests/test_underwriting.py tests/test_claims_fraud.py -v` (written in step 7) pass.

- [ ] 5. Write `src/underwriting.py`.
      `UnderwritingFeatures` and `UnderwritingResult` Pydantic v2 models. `UnderwritingEngine` trains XGBoost + LightGBM ensemble on `data/sample_policies.csv` (or synthetic fallback) with 10-trial Optuna tuning; `predict(features) -> UnderwritingResult`. Fallback chain: XGBoost/LightGBM → sklearn RandomForest → rule-based heuristic. Saves/loads model to `data/models/underwriting_model.pkl`. Uses `pathlib.Path` throughout.
      Files: `src/underwriting.py`
      Verify: `pytest tests/test_underwriting.py -v` — all tests pass.

- [ ] 6. Write `src/claims_fraud.py`.
      `ClaimFeatures` and `FraudScoringResult` Pydantic v2 models. `FraudDetectionEngine` — same ensemble + Optuna pattern trained on `data/sample_claims.csv`. Fallback chain identical to underwriting. Saves/loads to `data/models/fraud_model.pkl`.
      Files: `src/claims_fraud.py`
      Verify: `pytest tests/test_claims_fraud.py -v` — all tests pass.

- [ ] 7. Write `src/graph_collusion.py`.
      Bipartite graph claimants ↔ shared entities (repair shops, medical providers, witnesses). Degree + betweenness centrality; rings where ≥3 claims share ≥2 entities = suspicious. `CollusionRing` Pydantic v2 model. `GraphCollusionDetector.analyze(claims_df) -> list[CollusionRing]`. Primary: Neo4j driver; fallback: NetworkX; graceful: empty list on any import failure.
      Files: `src/graph_collusion.py`
      Verify: `pytest tests/ -v -k "not agent"` — no failures.

- [ ] 8. Write `tests/test_underwriting.py`, `tests/test_claims_fraud.py`.
      `test_underwriting.py`: test `predict()` returns `UnderwritingResult` with `risk_score` in [0,1] and `risk_tier` in {low/medium/high}; test fallback path when xgboost import is patched to fail.
      `test_claims_fraud.py`: test `predict()` returns `FraudScoringResult` with `fraud_score` in [0,1]; test fallback path.
      Files: `tests/test_underwriting.py`, `tests/test_claims_fraud.py`
      Verify: `pytest tests/test_underwriting.py tests/test_claims_fraud.py -v` — all pass.

---

## FEAT-003 — Security, Database, and Infra Modules

- [ ] 9. Write `src/encryption.py`.
      `encrypt_field(value: str) -> str`, `decrypt_field(encrypted: str) -> str` using Fernet. Key from `ENCRYPTION_KEY` env var or auto-generated and saved to `data/.encryption_key`. Uses `pathlib.Path`.
      Files: `src/encryption.py`
      Verify: `python -c "from src.encryption import encrypt_field, decrypt_field; assert decrypt_field(encrypt_field('test')) == 'test'"` exits 0.

- [ ] 10. Write `src/rbac.py`.
       Roles: admin, analyst, viewer. `RBACMiddleware` and `require_role(roles: list[str])` FastAPI dependency. API key → role mapping from `API_KEYS` env var (JSON dict) or hardcoded demo dict `{"demo-admin": "admin", "demo-analyst": "analyst", "demo-viewer": "viewer"}`.
       Files: `src/rbac.py`
       Verify: import succeeds — `python -c "from src.rbac import require_role"` exits 0.

- [ ] 11. Write `src/rate_limit.py`.
       `RateLimiter` class with `check_rate_limit(key, max_requests, window_seconds) -> bool` using in-memory sliding window. `rate_limit_dependency` FastAPI dependency factory.
       Files: `src/rate_limit.py`
       Verify: import succeeds — `python -c "from src.rate_limit import RateLimiter"` exits 0.

- [ ] 12. Write `src/database.py`.
       `DatabaseManager` async class. Primary: asyncpg (`DATABASE_URL`). Fallback 1: Supabase REST (`SUPABASE_URL` + `SUPABASE_KEY`). Fallback 2: CSV reads from `data/`. Methods: `get_claims(filters) -> list[dict]`, `get_policies(filters) -> list[dict]`, `save_decision(decision: dict)`. Each method catches connection errors and falls through the chain silently.
       Files: `src/database.py`
       Verify: `python -c "from src.database import DatabaseManager"` exits 0.

---

## FEAT-004 — RAG / Agent / Observability / Compliance Layer

- [ ] 13. Write `src/vector_store.py`.
       `PolicyVectorStore`: loads + chunks docs from `data/policy_docs/` (300-char chunks, 50-char overlap), embeds with `sentence-transformers/all-MiniLM-L6-v2` locally, persists to ChromaDB at `data/chroma_db/`. `search(query, top_k=3) -> list[dict]` returns `{content, source, score}`. Graceful: returns `[]` if `chromadb` or `sentence_transformers` not installed.
       Files: `src/vector_store.py`
       Verify: `python -c "from src.vector_store import PolicyVectorStore"` exits 0.

- [ ] 14. Write `src/hitl.py`.
       `HITLItem` Pydantic v2 model: `item_id`, `session_id`, `context_type`, `decision_draft`, `model_result`, `analyst_review` (Optional), `status` (pending/approved/rejected/escalated), `created_at`, `reviewed_at`. `HITLQueue`: in-memory list + `threading.Lock`, optional JSON persistence to `data/hitl_queue.json`. Methods: `enqueue(item)`, `get_pending() -> list[HITLItem]`, `review(item_id, decision, analyst_notes)`.
       Files: `src/hitl.py`
       Verify: `python -c "from src.hitl import HITLQueue, HITLItem"` exits 0.

- [ ] 15. Write `src/copilot_metrics.py`.
       Prometheus counters/histograms under `claimguard_copilot_*` namespace: `agent_runs_total` (label: `agent_name`), `agent_latency_seconds` (histogram, label: `agent_name`), `tool_calls_total` (label: `tool_name`), `retriever_hits_total`, `decisions_drafted_total`, `hitl_queue_depth` (gauge), `graph_queries_total`. Helper functions: `record_agent_run(agent_name, duration_seconds)`, `record_tool_call(tool_name)`, `record_retriever_hit()`, `record_decision_drafted()`, `set_hitl_queue_depth(n)`, `record_graph_query()`. All helpers are no-ops if `prometheus_client` not installed.
       Files: `src/copilot_metrics.py`
       Verify: `python -c "from src.copilot_metrics import record_agent_run; record_agent_run('test', 0.1)"` exits 0.

- [ ] 16. Write `src/agent_graph.py`.
       `CopilotState` TypedDict with all specified fields. Four node functions: `retriever_agent`, `tool_agent`, `writer_agent`, `hitl_router`. `WriterAgent` uses Groq `llama3-8b-8192` if `GROQ_API_KEY` set, else rule-based template. `HITLRouter` always sets `requires_human_review=True` and enqueues to `HITLQueue`. Build graph with LangGraph `StateGraph`; export `run_copilot(state) -> CopilotState` and `run_copilot_fallback(state) -> CopilotState` (linear function, no LangGraph). If `langgraph` not installed, `run_copilot` falls back to `run_copilot_fallback`.
       Files: `src/agent_graph.py`
       Verify: `pytest tests/test_agent_graph.py -v` — all pass.

- [ ] 17. Write `src/compliance_irdai.py`.
       10+ IRDAI controls: claim settlement 30 days, KYC, grievance redressal 15 days, policy issuance SLA, premium refund on cancellation, free-look period, portability rights, AML checks, fraud reporting to IRDAI, data localisation. `IRDAIControl` and `ComplianceReport` Pydantic v2 models. `generate_compliance_report() -> ComplianceReport` with mixed statuses (realistic demo).
       Files: `src/compliance_irdai.py`
       Verify: `pytest tests/test_compliance.py -v` — all pass.

- [ ] 18. Write `tests/test_agent_graph.py`, `tests/test_compliance.py`.
       `test_agent_graph.py`: test `run_copilot_fallback()` end-to-end, assert `requires_human_review=True` in result state.
       `test_compliance.py`: test `generate_compliance_report()` returns `ComplianceReport` with ≥5 controls and `overall_score` in [0,100].
       Files: `tests/test_agent_graph.py`, `tests/test_compliance.py`
       Verify: `pytest tests/ -v` — full suite passes.

---

## FEAT-005 — API, Streamlit UI, Evaluation, Docker, CI, README

- [ ] 19. Write `api/main.py`.
       FastAPI app with: X-API-Key auth via `rbac.py`, rate limiting via `rate_limit.py`, CORS, structured JSON logging. Routes: `POST /underwrite` (analyst/admin), `POST /claims/score` (analyst/admin), `POST /copilot/decide` (analyst/admin), `GET /graph/collusion-rings` (analyst/admin), `GET /compliance` (any auth), `GET /health` (public), `GET /metrics` (admin). Startup event initialises models and vector store lazily (non-blocking). Use `python-dotenv` to load `.env`.
       Files: `api/main.py`
       Verify: `python -c "from api.main import app"` exits 0 (import-only check, no server start).

- [ ] 20. Write `app/app.py`.
       Streamlit multi-tab app with 8 tabs: Explorer, Underwrite, Claims, Policy Copilot, Graph Intel, HITL, Observability, Compliance. Each tab wires to the corresponding `src/` module. SHAP waterfall chart via Plotly. Network graph in Graph Intel via Plotly. HITL tab shows pending queue, approve/reject/escalate buttons. Observability shows live Prometheus metrics or mock values. Compliance shows gauge + pie + expandable control cards.
       Files: `app/app.py`
       Verify: `python -c "import app.app"` exits 0 (import-only, no browser launch).

- [ ] 21. Write `eval/ragas_eval.py`.
       5 sample (question, context, answer) triples covering IRDAI claim settlement, coverage exclusions, claim procedures, portability, grievance redressal. RAGAS metrics: `context_precision`, `context_recall`, `faithfulness`, `answer_relevancy`. Graceful: prints mock score table if `ragas` not installed.
       Files: `eval/ragas_eval.py`
       Verify: `python eval/ragas_eval.py` exits 0 (with or without ragas installed).

- [ ] 22. Write `Dockerfile` (multi-stage: builder installs deps, final copies src; exposes 8000 + 8501) and `docker-compose.yml` (services: api, frontend, postgres:15, neo4j:5, prometheus, grafana; with env vars, volumes, health checks).
       Files: `Dockerfile`, `docker-compose.yml`
       Verify: `docker compose config` exits 0 (validates YAML without starting).

- [ ] 23. Write `.github/workflows/ci.yml`.
       Triggers: push/PR to main. Steps: checkout → Python 3.11 → `pip install -r requirements.txt` → `python data/synthetic_generator.py` → `pytest tests/ -v --tb=short`.
       Files: `.github/workflows/ci.yml`
       Verify: file is valid YAML (parse check) and contains correct pytest command.

- [ ] 24. Write comprehensive `README.md`: overview, ASCII architecture diagram, quick start (Docker + local), env vars table, API reference, UI tabs description, compliance summary, evaluation, tech stack badges.
       Files: `README.md`
       Verify: file exists and is >2 KB.

- [ ] 25. Final integration check: run `python data/synthetic_generator.py && pytest tests/ -v --tb=short` — full test suite passes with no errors.

---

## Dependency map

```
FEAT-001 (data)  → FEAT-002 (ML models read CSVs)
FEAT-002 (ML)    → FEAT-004 (agent calls models; tests/test_agent_graph uses run_copilot_fallback)
FEAT-003 (infra) → FEAT-004 (agent_graph imports hitl; api imports rbac/rate_limit)
FEAT-004 (agent) → FEAT-005 (api/main and app/app import all src modules)
FEAT-001–004     → FEAT-005 (README, Docker, CI reference all components)
```
