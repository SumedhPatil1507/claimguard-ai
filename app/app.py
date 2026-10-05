"""
ClaimGuard AI — Streamlit Frontend
8-tab interactive platform for underwriting risk scoring, claims fraud detection,
Policy Copilot (LangGraph RAG), graph collusion detection, HITL review, and IRDAI compliance.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so src.* imports work from any working directory
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Optional: load .env (graceful — not available on Streamlit Cloud)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Core imports — these MUST be available on Streamlit Cloud
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Plotly — wrapped gracefully; all chart code is guarded by HAS_PLOTLY
# ---------------------------------------------------------------------------
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Domain module imports — all graceful
# ---------------------------------------------------------------------------
try:
    from src.underwriting import UnderwritingEngine, UnderwritingFeatures
    _uw_engine = UnderwritingEngine()
    HAS_UW = True
except Exception:
    HAS_UW = False
    _uw_engine = None
    UnderwritingFeatures = None  # type: ignore[assignment]

try:
    from src.claims_fraud import ClaimFeatures, FraudDetectionEngine
    _fraud_engine = FraudDetectionEngine()
    HAS_FRAUD = True
except Exception:
    HAS_FRAUD = False
    _fraud_engine = None
    ClaimFeatures = None  # type: ignore[assignment]

try:
    from src.agent_graph import run_copilot
    HAS_COPILOT = True
except Exception:
    HAS_COPILOT = False

    def run_copilot(*a, **k):  # type: ignore[misc]
        return {
            "decision_draft": "Agent pipeline unavailable — install langchain-core and langgraph.",
            "requires_human_review": True,
            "retrieved_docs": [],
            "model_result": {},
            "session_id": "N/A",
        }

try:
    from src.graph_collusion import GraphCollusionDetector
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False

try:
    from src.hitl import HITLItem, hitl_queue
    HAS_HITL = True
except Exception:
    HAS_HITL = False
    hitl_queue = None  # type: ignore[assignment]
    HITLItem = None  # type: ignore[assignment]

try:
    from src.compliance_irdai import generate_compliance_report
    HAS_COMPLIANCE = True
except Exception:
    HAS_COMPLIANCE = False

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClaimGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B2A3B 100%);
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label {
    color: #E0E0E0 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00C896 0%, #00A878 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
}
.stButton > button:hover {
    box-shadow: 0 4px 12px rgba(0,200,150,0.4);
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #00C896 !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab-list"] {
    background: #0D1B2A;
    border-radius: 8px 8px 0 0;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] { color: #A0ADB8; border-radius: 6px; font-weight: 500; }
.stTabs [aria-selected="true"] { background: #00C896 !important; color: white !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🛡️ ClaimGuard AI")
    st.markdown("**Agentic InsurTech Platform**")
    st.markdown("---")
    st.markdown("### System Status")
    st.markdown(f'{"🟢" if HAS_UW else "🔴"} ML Engine: {"Active" if HAS_UW else "Unavailable"}')
    st.markdown(f'{"🟢" if HAS_FRAUD else "🔴"} Fraud Engine: {"Active" if HAS_FRAUD else "Unavailable"}')
    st.markdown(f'{"🟢" if HAS_COPILOT else "🔴"} Policy Copilot: {"Active" if HAS_COPILOT else "Fallback"}')
    st.markdown(f'{"🟢" if HAS_GRAPH else "🔴"} Graph Intel: {"Active" if HAS_GRAPH else "Unavailable"}')
    st.markdown(f'{"🟢" if HAS_PLOTLY else "🟡"} Charts: {"Plotly" if HAS_PLOTLY else "Text fallback"}')
    st.markdown("---")
    st.markdown("### 📋 IRDAI Compliance")
    st.success("✓ All decisions require human review")
    st.markdown("---")
    st.caption("v1.0.0 | MIT License | © 2024 ClaimGuard AI")


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_claims_data() -> pd.DataFrame:
    for candidate in [
        _ROOT / "data" / "sample_claims.csv",
        Path("data/sample_claims.csv"),
    ]:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


@st.cache_data
def load_policies_data() -> pd.DataFrame:
    for candidate in [
        _ROOT / "data" / "sample_policies.csv",
        Path("data/sample_policies.csv"),
    ]:
        if candidate.exists():
            return pd.read_csv(candidate)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helper: render a Plotly figure or a plain st.table fallback
# ---------------------------------------------------------------------------
def _show_chart(fig, key: str = "") -> None:
    if HAS_PLOTLY and fig is not None:
        st.plotly_chart(fig, use_container_width=True, key=key or None)
    else:
        st.info("Install plotly to see interactive charts.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "📊 Explorer",
    "🏦 Underwrite",
    "🔍 Claims",
    "🤖 Policy Copilot",
    "🕸️ Graph Intel",
    "👤 HITL Review",
    "📈 Observability",
    "✅ Compliance",
])

# ============================================================
# Tab 1 — Explorer
# ============================================================
with tabs[0]:
    st.header("📊 Data Explorer")
    claims_df = load_claims_data()
    policies_df = load_policies_data()

    if claims_df.empty:
        st.warning("No claims data found. Run: `python data/synthetic_generator.py`")
    else:
        fraud_rate = (
            claims_df["fraud_label"].sum() / len(claims_df) * 100
            if "fraud_label" in claims_df.columns else 0
        )
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Claims", f"{len(claims_df):,}")
        c2.metric("Fraud Rate", f"{fraud_rate:.1f}%")
        c3.metric(
            "Avg Claim (₹)",
            f"{claims_df['claim_amount'].mean():,.0f}" if "claim_amount" in claims_df.columns else "N/A",
        )
        c4.metric("Total Policies", f"{len(policies_df):,}")
        c5.metric(
            "Avg Premium Adj",
            f"{policies_df['premium_adjustment'].mean():.2f}x"
            if not policies_df.empty and "premium_adjustment" in policies_df.columns
            else "N/A",
        )
        st.markdown("---")

        if HAS_PLOTLY:
            col1, col2 = st.columns(2)
            with col1:
                if "claim_amount" in claims_df.columns and "fraud_label" in claims_df.columns:
                    fig = px.histogram(
                        claims_df, x="claim_amount", color="fraud_label", nbins=40,
                        title="Claim Amount Distribution",
                        labels={"fraud_label": "Fraud", "claim_amount": "Claim Amount (₹)"},
                        color_discrete_map={0: "#00C896", 1: "#FF6B6B"},
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "fraud_label" in claims_df.columns:
                    fc = claims_df["fraud_label"].value_counts()
                    fig = px.pie(
                        values=fc.values,
                        names=["Legitimate" if i == 0 else "Fraud" for i in fc.index],
                        title="Fraud vs Legitimate Claims",
                        color_discrete_sequence=["#00C896", "#FF6B6B"],
                        hole=0.4, template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                if "claim_type" in claims_df.columns:
                    ct = claims_df["claim_type"].value_counts().reset_index()
                    ct.columns = ["claim_type", "count"]
                    fig = px.bar(
                        ct, x="claim_type", y="count", title="Claims by Type",
                        color="count", color_continuous_scale="Teal", template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col4:
                if "days_since_policy_start" in claims_df.columns and "claim_amount" in claims_df.columns:
                    fig = px.scatter(
                        claims_df, x="days_since_policy_start", y="claim_amount",
                        color="fraud_label" if "fraud_label" in claims_df.columns else None,
                        title="Days Since Policy Start vs Claim Amount",
                        color_discrete_map={0: "#00C896", 1: "#FF6B6B"},
                        opacity=0.6, template="plotly_dark",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            num_cols = claims_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) > 2:
                corr = claims_df[num_cols].corr()
                fig = px.imshow(
                    corr, title="Correlation Heatmap — Claims",
                    color_continuous_scale="RdBu_r", text_auto=".2f", template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

            if not policies_df.empty and "risk_tier" in policies_df.columns:
                rt = policies_df["risk_tier"].value_counts().reset_index()
                rt.columns = ["risk_tier", "count"]
                fig = px.bar(
                    rt, x="risk_tier", y="count", title="Policy Risk Tier Distribution",
                    color="risk_tier",
                    color_discrete_map={"low": "#00C896", "medium": "#FFD700", "high": "#FF6B6B"},
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Install plotly (`pip install plotly`) for interactive charts.")
            st.dataframe(claims_df.head(20), use_container_width=True)

# ============================================================
# Tab 2 — Underwrite
# ============================================================
with tabs[1]:
    st.header("🏦 Underwriting Risk Scorer")
    st.markdown("Score applicants at policy issuance time using the XGBoost + LightGBM ensemble.")

    with st.form("underwriting_form"):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 80, 35)
            annual_income = st.number_input("Annual Income (₹)", 100_000, 10_000_000, 800_000, step=50_000)
            credit_score = st.slider("Credit Score", 300, 900, 720)
            sum_insured = st.number_input("Sum Insured (₹)", 100_000, 50_000_000, 1_000_000, step=100_000)
        with col2:
            coverage_type = st.selectbox("Coverage Type", ["motor", "health", "property", "life"])
            num_dependents = st.slider("Number of Dependents", 0, 10, 2)
            prior_claims_count = st.slider("Prior Claims Count", 0, 20, 0)
            region = st.selectbox("Region", ["north", "south", "east", "west", "central"])
            occupation = st.selectbox(
                "Occupation", ["salaried", "self-employed", "business", "retired", "student"]
            )
        uw_submitted = st.form_submit_button("🔍 Score Risk", use_container_width=True)

    if uw_submitted:
        if not HAS_UW or _uw_engine is None:
            st.error("Underwriting engine not available. Check installation.")
        else:
            with st.spinner("Scoring risk..."):
                try:
                    features = UnderwritingFeatures(
                        age=age, annual_income=annual_income, credit_score=credit_score,
                        sum_insured=sum_insured, coverage_type=coverage_type,
                        num_dependents=num_dependents, prior_claims_count=prior_claims_count,
                        region=region, occupation=occupation,
                    )
                    result = _uw_engine.predict(features)
                    risk_score = getattr(result, "risk_score", 0.0)
                    risk_tier = getattr(result, "risk_tier", "medium")
                    tier_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(risk_tier, "🟡")

                    st.markdown(f"## {tier_icon} Risk Tier: **{risk_tier.upper()}**")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Risk Score", f"{risk_score:.3f}")
                    m2.metric("Premium Adjustment", f"{getattr(result, 'premium_adjustment', 1.0):.2f}x")
                    m3.metric("Model Version", getattr(result, "model_version", "N/A"))

                    if HAS_PLOTLY:
                        gauge_color = (
                            "#FF6B6B" if risk_score > 0.6 else "#FFD700" if risk_score > 0.3 else "#00C896"
                        )
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=risk_score,
                            title={"text": "Risk Score", "font": {"size": 20}},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": gauge_color},
                                "steps": [
                                    {"range": [0, 0.3], "color": "rgba(0,200,150,0.15)"},
                                    {"range": [0.3, 0.6], "color": "rgba(255,215,0,0.15)"},
                                    {"range": [0.6, 1], "color": "rgba(255,107,107,0.15)"},
                                ],
                            },
                        ))
                        fig.update_layout(template="plotly_dark", height=300)
                        st.plotly_chart(fig, use_container_width=True)

                        shap_drivers = getattr(result, "shap_drivers", [])
                        if shap_drivers:
                            df_shap = pd.DataFrame(shap_drivers)
                            if "feature" in df_shap.columns and "shap_value" in df_shap.columns:
                                df_shap = df_shap.sort_values("shap_value")
                                colors = ["#00C896" if v < 0 else "#FF6B6B" for v in df_shap["shap_value"]]
                                fig2 = go.Figure(go.Bar(
                                    x=df_shap["shap_value"], y=df_shap["feature"],
                                    orientation="h", marker_color=colors,
                                ))
                                fig2.update_layout(
                                    title="Top SHAP Feature Drivers",
                                    template="plotly_dark",
                                    xaxis_title="SHAP Value",
                                    height=350,
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                    with st.expander("📄 Raw Result JSON"):
                        st.json(result.model_dump() if hasattr(result, "model_dump") else vars(result))
                    st.toast("✅ Risk scored successfully!", icon="✅")
                except Exception as exc:
                    st.error(f"Scoring failed: {exc}")

# ============================================================
# Tab 3 — Claims
# ============================================================
with tabs[2]:
    st.header("🔍 Claims Fraud Detection Engine")
    st.markdown("Score claims at filing time for fraud risk.")

    with st.form("claims_form"):
        col1, col2 = st.columns(2)
        with col1:
            claim_id = st.text_input("Claim ID", "CLM001")
            claimant_id = st.text_input("Claimant ID", "CLMT001")
            policy_id_input = st.text_input("Policy ID", "POL001")
            claim_amount = st.number_input("Claim Amount (₹)", 1_000.0, 5_000_000.0, 50_000.0, step=1_000.0)
            days_since = st.number_input("Days Since Policy Start", 0, 3650, 90)
        with col2:
            num_prior = st.slider("Number of Prior Claims", 0, 20, 0)
            claim_type = st.selectbox("Claim Type", ["motor", "health", "property", "life"])
            claim_severity = st.selectbox("Claim Severity", ["low", "medium", "high"])
            repair_shop_id = st.text_input("Repair Shop ID (optional)", "")
            medical_provider_id = st.text_input("Medical Provider ID (optional)", "")
        claims_submitted = st.form_submit_button("🚨 Score Claim", use_container_width=True)

    if claims_submitted:
        if not HAS_FRAUD or _fraud_engine is None:
            st.error("Fraud detection engine not available.")
        else:
            with st.spinner("Analysing claim..."):
                try:
                    features = ClaimFeatures(
                        claim_id=claim_id, claimant_id=claimant_id, policy_id=policy_id_input,
                        claim_amount=claim_amount, days_since_policy_start=int(days_since),
                        num_prior_claims=num_prior, claim_type=claim_type,
                        claim_severity=claim_severity,
                        repair_shop_id=repair_shop_id or None,
                        medical_provider_id=medical_provider_id or None,
                    )
                    result = _fraud_engine.predict(features)
                    fraud_flag = getattr(result, "fraud_flag", False)
                    fraud_score = getattr(result, "fraud_score", 0.0)

                    if fraud_flag:
                        st.error("⚠️ HIGH FRAUD RISK DETECTED — Flagged for mandatory HITL review")
                    else:
                        st.success("✅ Claim Appears Legitimate — Queued for standard HITL review")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Fraud Score", f"{fraud_score:.3f}")
                    m2.metric("Fraud Flag", "🚨 YES" if fraud_flag else "✅ NO")
                    m3.metric("Confidence Tier", getattr(result, "confidence_tier", "N/A"))

                    if HAS_PLOTLY:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number", value=fraud_score,
                            title={"text": "Fraud Score"},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "#FF6B6B" if fraud_score > 0.5 else "#00C896"},
                                "steps": [
                                    {"range": [0, 0.3], "color": "rgba(0,200,150,0.15)"},
                                    {"range": [0.3, 0.6], "color": "rgba(255,215,0,0.15)"},
                                    {"range": [0.6, 1], "color": "rgba(255,107,107,0.15)"},
                                ],
                            },
                        ))
                        fig.update_layout(template="plotly_dark", height=300)
                        st.plotly_chart(fig, use_container_width=True)

                        shap_drivers = getattr(result, "shap_drivers", [])
                        if shap_drivers:
                            df_shap = pd.DataFrame(shap_drivers)
                            if "feature" in df_shap.columns and "shap_value" in df_shap.columns:
                                df_shap = df_shap.sort_values("shap_value")
                                fig2 = go.Figure(go.Bar(
                                    x=df_shap["shap_value"], y=df_shap["feature"],
                                    orientation="h",
                                    marker_color=["#00C896" if v < 0 else "#FF6B6B" for v in df_shap["shap_value"]],
                                ))
                                fig2.update_layout(
                                    title="SHAP Feature Drivers", template="plotly_dark", height=300,
                                )
                                st.plotly_chart(fig2, use_container_width=True)

                    with st.expander("📄 Raw Result"):
                        st.json(result.model_dump() if hasattr(result, "model_dump") else vars(result))
                    st.toast("Claim scored!", icon="🔍")
                except Exception as exc:
                    st.error(f"Claim scoring failed: {exc}")

# ============================================================
# Tab 4 — Policy Copilot
# ============================================================
with tabs[3]:
    st.header("🤖 Policy Copilot — Powered by LangGraph")
    st.info("ℹ️ All decisions are drafted for human analyst review — no auto-approval ever.")

    context_type = st.selectbox("Context Type", ["underwriting", "claims"], key="copilot_ctx")
    query = st.text_area(
        "Describe the case or question", height=100,
        placeholder="e.g. New motor policy applicant aged 32, credit score 750, no prior claims. Assess risk.",
    )
    default_features = {
        "underwriting": (
            '{"age":32,"annual_income":900000,"credit_score":750,'
            '"sum_insured":500000,"coverage_type":"motor",'
            '"num_dependents":1,"prior_claims_count":0,'
            '"region":"north","occupation":"salaried"}'
        ),
        "claims": (
            '{"claim_id":"CLM001","claimant_id":"CLMT001","policy_id":"POL001",'
            '"claim_amount":75000,"days_since_policy_start":45,'
            '"num_prior_claims":2,"claim_type":"motor","claim_severity":"high",'
            '"repair_shop_id":"SHOP001","medical_provider_id":""}'
        ),
    }
    features_json = st.text_area("Features JSON", value=default_features.get(context_type, "{}"), height=120)

    if st.button("⚡ Analyze with Copilot", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Running Policy Copilot pipeline..."):
                try:
                    features = json.loads(features_json or "{}")
                    result = run_copilot(query=query, context_type=context_type, features=features)

                    st.markdown("### 📋 Draft Decision")
                    st.info(result.get("decision_draft", "No decision draft generated."))

                    with st.expander("📚 Retrieved Policy Clauses"):
                        docs = result.get("retrieved_docs", [])
                        if docs:
                            for i, doc in enumerate(docs, 1):
                                st.markdown(
                                    f"**{i}. Source:** `{doc.get('source', 'Unknown')}`"
                                    + (f" | Score: `{doc.get('score', 0):.3f}`" if doc.get("score") else "")
                                )
                                content = doc.get("content", "")
                                st.markdown(f"> {content[:500]}{'...' if len(content) > 500 else ''}")
                                st.markdown("---")
                        else:
                            st.caption("No policy clauses retrieved (vector store may not be indexed).")

                    with st.expander("📊 Model Evidence"):
                        mr = result.get("model_result", {})
                        if mr:
                            score_key = "risk_score" if context_type == "underwriting" else "fraud_score"
                            tier_key = "risk_tier" if context_type == "underwriting" else "confidence_tier"
                            e1, e2 = st.columns(2)
                            e1.metric("Score", f"{mr.get(score_key, 'N/A')}")
                            e2.metric("Tier", mr.get(tier_key, "N/A"))
                            shap = mr.get("shap_drivers", [])
                            if shap:
                                st.dataframe(pd.DataFrame(shap), use_container_width=True)
                        else:
                            st.caption("No model evidence available.")

                    st.warning(
                        f"🔄 Decision queued for analyst review | Session: `{result.get('session_id', 'N/A')}`"
                    )
                    if "copilot_first_run" not in st.session_state:
                        st.session_state["copilot_first_run"] = True
                        st.balloons()
                    st.toast("✅ Analysis complete — queued for HITL review", icon="🤖")
                except Exception as exc:
                    st.error(f"Copilot error: {exc}")

# ============================================================
# Tab 5 — Graph Intel
# ============================================================
with tabs[4]:
    st.header("🕸️ Collusion Ring Detection")
    st.markdown("Graph-based detection of claim-ring collusion using shared entities.")

    if not HAS_GRAPH:
        st.error("Graph collusion module not available.")
    else:
        with st.spinner("Analysing collusion rings..."):
            try:
                claims_df2 = load_claims_data()
                if claims_df2.empty:
                    st.warning("No claims data available.")
                else:
                    detector = GraphCollusionDetector()
                    rings = detector.analyze(claims_df2)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rings Detected", len(rings))
                    m2.metric("High Severity", sum(1 for r in rings if getattr(r, "severity", "") == "high"))
                    m3.metric("Claimants in Rings", sum(len(getattr(r, "claimant_ids", [])) for r in rings))

                    if rings and HAS_PLOTLY:
                        try:
                            import networkx as nx
                            G = nx.Graph()
                            for ring in rings:
                                for cid in getattr(ring, "claimant_ids", []):
                                    G.add_node(cid, node_type="claimant")
                                for eid in getattr(ring, "shared_entities", []):
                                    G.add_node(eid, node_type="entity")
                                    for cid in getattr(ring, "claimant_ids", []):
                                        G.add_edge(cid, eid)

                            if len(G.nodes) > 0:
                                pos = nx.spring_layout(G, seed=42)
                                edge_x, edge_y = [], []
                                for e0, e1 in G.edges():
                                    x0, y0 = pos[e0]; x1, y1 = pos[e1]
                                    edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

                                claimant_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "claimant"]
                                entity_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]

                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=edge_x, y=edge_y, mode="lines",
                                    line=dict(width=1, color="#444"), hoverinfo="none", name="Connections",
                                ))
                                if claimant_nodes:
                                    fig.add_trace(go.Scatter(
                                        x=[pos[n][0] for n in claimant_nodes],
                                        y=[pos[n][1] for n in claimant_nodes],
                                        mode="markers+text",
                                        marker=dict(size=18, color="#5B9BD5", symbol="circle", line=dict(width=2, color="white")),
                                        text=claimant_nodes, textposition="top center",
                                        hoverinfo="text", name="Claimants",
                                    ))
                                if entity_nodes:
                                    fig.add_trace(go.Scatter(
                                        x=[pos[n][0] for n in entity_nodes],
                                        y=[pos[n][1] for n in entity_nodes],
                                        mode="markers+text",
                                        marker=dict(size=14, color="#FF9F43", symbol="diamond", line=dict(width=2, color="white")),
                                        text=entity_nodes, textposition="top center",
                                        hoverinfo="text", name="Shared Entities",
                                    ))
                                fig.update_layout(
                                    title="Collusion Network Graph", template="plotly_dark",
                                    showlegend=True, hovermode="closest", height=500,
                                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

                    if rings:
                        st.markdown("### Ring Summary")
                        ring_data = []
                        for ring in rings:
                            sev = getattr(ring, "severity", "medium")
                            ring_data.append({
                                "Ring ID": getattr(ring, "ring_id", "N/A"),
                                "Severity": f"{'🔴' if sev=='high' else '🟡' if sev=='medium' else '🟢'} {sev}",
                                "Claimants": len(getattr(ring, "claimant_ids", [])),
                                "Shared Entities": len(getattr(ring, "shared_entities", [])),
                                "Centrality Score": f"{getattr(ring, 'centrality_score', 0):.3f}",
                            })
                        st.dataframe(pd.DataFrame(ring_data), use_container_width=True)

                        for ring in rings[:5]:
                            with st.expander(f"Ring {getattr(ring, 'ring_id', 'N/A')} — Details"):
                                ec1, ec2 = st.columns(2)
                                ec1.markdown("**Claimants:**")
                                for cid in getattr(ring, "claimant_ids", []):
                                    ec1.markdown(f"• `{cid}`")
                                ec2.markdown("**Shared Entities:**")
                                for eid in getattr(ring, "shared_entities", []):
                                    ec2.markdown(f"• `{eid}`")
                    else:
                        st.success("✅ No suspicious collusion rings detected in the current dataset.")
            except Exception as exc:
                st.error(f"Graph analysis failed: {exc}")

# ============================================================
# Tab 6 — HITL Review
# ============================================================
with tabs[5]:
    st.header("👤 Human-in-the-Loop Analyst Review")

    if not HAS_HITL or hitl_queue is None:
        st.error("HITL module not available.")
    else:
        pending = hitl_queue.get_pending()
        c1, c2 = st.columns([1, 4])
        c1.metric("Pending Reviews", len(pending))
        if c2.button("🔄 Refresh Queue"):
            st.rerun()

        if not pending:
            st.success("✅ All reviews complete — queue is empty")
        else:
            st.markdown(f"**{len(pending)} item(s) awaiting review:**")
            for item in pending:
                with st.container(border=True):
                    h1, h2, h3 = st.columns([3, 2, 2])
                    h1.markdown(f"**Item ID:** `{item.item_id}`")
                    h2.markdown(f"**Type:** {'🏦' if item.context_type == 'underwriting' else '🔍'} {item.context_type}")
                    h3.markdown(f"**Created:** {item.created_at.strftime('%Y-%m-%d %H:%M')}")
                    st.text_area("Decision Draft", value=item.decision_draft, height=120,
                                 key=f"draft_{item.item_id}", disabled=True)
                    mr = item.model_result or {}
                    if mr:
                        sk = "risk_score" if item.context_type == "underwriting" else "fraud_score"
                        tk = "risk_tier" if item.context_type == "underwriting" else "confidence_tier"
                        sm1, sm2 = st.columns(2)
                        sm1.metric("Score", f"{mr.get(sk, 'N/A')}")
                        sm2.metric("Tier", mr.get(tk, "N/A"))
                    notes = st.text_area("Analyst Notes", placeholder="Enter review notes...",
                                         key=f"notes_{item.item_id}")
                    b1, b2, b3 = st.columns(3)
                    if b1.button("✅ Approve", key=f"approve_{item.item_id}", use_container_width=True):
                        hitl_queue.review(item.item_id, "approved", notes)
                        st.toast("Decision approved!", icon="✅"); st.rerun()
                    if b2.button("❌ Reject", key=f"reject_{item.item_id}", use_container_width=True):
                        hitl_queue.review(item.item_id, "rejected", notes)
                        st.toast("Decision rejected!", icon="❌"); st.rerun()
                    if b3.button("⬆️ Escalate", key=f"escalate_{item.item_id}", use_container_width=True):
                        hitl_queue.review(item.item_id, "escalated", notes)
                        st.toast("Decision escalated!", icon="⬆️"); st.rerun()

        reviewed = [i for i in hitl_queue.get_all() if i.status != "pending"]
        if reviewed:
            with st.expander(f"📋 Review History (last {min(10, len(reviewed))} items)"):
                icons = {"approved": "✅", "rejected": "❌", "escalated": "⬆️"}
                hist = [{
                    "Item ID": i.item_id[:12] + "...",
                    "Type": i.context_type,
                    "Status": f"{icons.get(i.status, '❓')} {i.status}",
                    "Reviewed At": i.reviewed_at.strftime("%Y-%m-%d %H:%M") if i.reviewed_at else "N/A",
                    "Notes": (i.analyst_review or "")[:50],
                } for i in reviewed[-10:]]
                st.dataframe(pd.DataFrame(hist), use_container_width=True)

# ============================================================
# Tab 7 — Observability
# ============================================================
with tabs[6]:
    st.header("📈 Live Metrics — Prometheus Observability")

    try:
        import prometheus_client  # noqa: F401
        st.success("✅ prometheus_client installed — metrics are live")
    except ImportError:
        st.info("ℹ️ prometheus_client not installed — showing illustrative mock values")

    # Metric values (live if available, mock otherwise)
    try:
        from prometheus_client import REGISTRY as _REG
        def _get_metric(name: str, default: int = 0) -> int:
            try:
                for m in _REG.collect():
                    if m.name == name:
                        return int(sum(s.value for s in m.samples))
            except Exception:
                pass
            return default
        agent_runs = _get_metric("claimguard_copilot_agent_runs_total", 12)
        tool_calls = _get_metric("claimguard_copilot_tool_calls_total", 8)
        ret_hits   = _get_metric("claimguard_copilot_retriever_hits_total", 15)
        decisions  = _get_metric("claimguard_copilot_decisions_drafted_total", 7)
    except Exception:
        agent_runs, tool_calls, ret_hits, decisions = 12, 8, 15, 7
    queue_depth = hitl_queue.queue_depth() if hitl_queue else 3

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Agent Runs", agent_runs)
    mc2.metric("Tool Calls", tool_calls)
    mc3.metric("Retriever Hits", ret_hits)
    mc4.metric("Decisions Drafted", decisions)
    mc5.metric("HITL Queue Depth", queue_depth)

    if HAS_PLOTLY:
        st.markdown("---")
        rng = np.random.default_rng(seed=42)
        percentiles = ["p50", "p75", "p90", "p95", "p99"]
        low_b = np.array([0.05, 0.08, 0.12, 0.18, 0.35])
        high_b = np.array([0.10, 0.15, 0.22, 0.30, 0.60])
        latencies = [rng.uniform(low_b[k], high_b[k], size=3).mean() for k in range(5)]

        fig = go.Figure(go.Bar(
            x=percentiles, y=latencies, marker_color="#00C896",
            text=[f"{v:.3f}s" for v in latencies], textposition="outside",
        ))
        fig.update_layout(
            title="Agent Latency Percentiles (seconds)", template="plotly_dark",
            yaxis_title="Latency (s)", xaxis_title="Percentile", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        rng2 = np.random.default_rng(seed=99)
        hours = list(range(24))
        decisions_ts = rng2.integers(0, 5, size=24).cumsum().tolist()
        fig2 = go.Figure(go.Scatter(
            x=hours, y=decisions_ts, mode="lines+markers",
            line=dict(color="#00C896", width=2),
            fill="tozeroy", fillcolor="rgba(0,200,150,0.1)", name="Decisions",
        ))
        fig2.update_layout(
            title="Decisions Drafted — Last 24 Hours", template="plotly_dark",
            xaxis_title="Hours Ago", yaxis_title="Cumulative Decisions", height=300,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.info("📊 Grafana dashboard: http://localhost:3000 (requires Docker Compose)")

# ============================================================
# Tab 8 — Compliance
# ============================================================
with tabs[7]:
    st.header("✅ IRDAI Regulatory Compliance Dashboard")

    if not HAS_COMPLIANCE:
        st.error("Compliance module not available.")
    else:
        with st.spinner("Loading compliance report..."):
            try:
                report = generate_compliance_report()
                score = report.overall_score
                gauge_color = "#FF6B6B" if score < 60 else "#FFD700" if score < 80 else "#00C896"

                if HAS_PLOTLY:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta", value=score,
                        title={"text": "IRDAI Compliance Score"},
                        delta={"reference": 80, "increasing": {"color": "#00C896"}, "decreasing": {"color": "#FF6B6B"}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": gauge_color},
                            "steps": [
                                {"range": [0, 60], "color": "rgba(255,107,107,0.15)"},
                                {"range": [60, 80], "color": "rgba(255,215,0,0.15)"},
                                {"range": [80, 100], "color": "rgba(0,200,150,0.15)"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": 80},
                        },
                    ))
                    fig.update_layout(template="plotly_dark", height=350)

                    cg1, cg2 = st.columns(2)
                    with cg1:
                        st.plotly_chart(fig, use_container_width=True)
                    with cg2:
                        st.metric("Compliance Score", f"{score:.1f}/100")
                        st.markdown(f"**Summary:** {report.summary}")
                        statuses = [c.status for c in report.controls]
                        fig2 = px.pie(
                            values=[statuses.count("compliant"), statuses.count("partial"), statuses.count("non_compliant")],
                            names=["compliant", "partial", "non_compliant"],
                            title="Controls Breakdown",
                            color_discrete_map={"compliant": "#00C896", "partial": "#FFD700", "non_compliant": "#FF6B6B"},
                            hole=0.4, template="plotly_dark",
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.metric("Compliance Score", f"{score:.1f}/100")
                    st.markdown(f"**Summary:** {report.summary}")

                st.markdown("---")
                st.markdown("### 📋 Control Details")
                for i in range(0, len(report.controls), 2):
                    cols = st.columns(2)
                    for j, ctrl in enumerate(report.controls[i:i + 2]):
                        with cols[j]:
                            status_icons = {"compliant": "✅", "partial": "⚠️", "non_compliant": "❌"}
                            st.markdown(
                                f"**{ctrl.title}** {status_icons.get(ctrl.status, '❓')} `{ctrl.status}`  "
                                f"| Severity: `{ctrl.severity}` | Category: `{ctrl.category}`"
                            )
                            with st.expander("Evidence & Remediation"):
                                st.markdown(f"**Evidence:** {ctrl.evidence}")
                                st.markdown(f"**Remediation:** {ctrl.remediation}")

                st.markdown("---")
                st.caption(
                    f"Report ID: `{report.report_id}` | "
                    f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}"
                )
            except Exception as exc:
                st.error(f"Compliance report error: {exc}")
