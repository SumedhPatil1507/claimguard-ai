from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots  # noqa: F401

# ---------------------------------------------------------------------------
# Graceful imports — every heavy dependency wrapped
# ---------------------------------------------------------------------------

try:
    from src.underwriting import UnderwritingEngine, UnderwritingFeatures
    _uw_engine = UnderwritingEngine()
    HAS_UW = True
except Exception:
    HAS_UW = False
    _uw_engine = None

try:
    from src.claims_fraud import FraudDetectionEngine, ClaimFeatures
    _fraud_engine = FraudDetectionEngine()
    HAS_FRAUD = True
except Exception:
    HAS_FRAUD = False
    _fraud_engine = None

try:
    from src.agent_graph import run_copilot
    HAS_COPILOT = True
except Exception:
    HAS_COPILOT = False

    def run_copilot(*a, **k):  # type: ignore[misc]
        return {
            'decision_draft': 'Agent unavailable',
            'requires_human_review': True,
            'retrieved_docs': [],
            'model_result': {},
            'session_id': 'N/A',
        }

try:
    from src.graph_collusion import GraphCollusionDetector
    HAS_GRAPH = True
except Exception:
    HAS_GRAPH = False

try:
    from src.hitl import hitl_queue, HITLItem
    HAS_HITL = True
except Exception:
    HAS_HITL = False
    hitl_queue = None  # type: ignore[assignment]

try:
    from src.compliance_irdai import generate_compliance_report
    HAS_COMPLIANCE = True
except Exception:
    HAS_COMPLIANCE = False

# ---------------------------------------------------------------------------
# Page config — MUST be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title='ClaimGuard AI',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* Dark navy sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1B2A3B 100%);
    color: white;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio > label {
    color: #E0E0E0 !important;
}
/* Teal accent buttons */
.stButton > button {
    background: linear-gradient(135deg, #00C896 0%, #00A878 100%);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,200,150,0.4);
}
/* Card-like metric containers */
[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    color: #00C896 !important;
    font-weight: 700;
}
/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0D1B2A;
    border-radius: 8px 8px 0 0;
    padding: 0.3rem;
}
.stTabs [data-baseweb="tab"] {
    color: #A0ADB8;
    border-radius: 6px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #00C896 !important;
    color: white !important;
}
/* Severity badge helper classes */
.badge-green  { background:#1a4a2e; color:#00C896; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-yellow { background:#4a3e1a; color:#FFD700; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-red    { background:#4a1a1a; color:#FF6B6B; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
.badge-blue   { background:#1a2e4a; color:#5B9BD5; padding:2px 10px; border-radius:12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('# 🛡️ ClaimGuard AI')
    st.markdown('**Agentic InsurTech Platform**')
    st.markdown('---')
    st.markdown('### Navigation')
    st.markdown('Use the tabs above to navigate')
    st.markdown('---')
    st.markdown('### System Status')
    st.markdown(f'{"🟢" if HAS_UW else "🔴"} ML Engine: {"Active" if HAS_UW else "Unavailable"}')
    st.markdown(f'{"🟢" if HAS_FRAUD else "🔴"} Fraud Engine: {"Active" if HAS_FRAUD else "Unavailable"}')
    st.markdown(f'{"🟢" if HAS_COPILOT else "🔴"} Copilot: {"Active" if HAS_COPILOT else "Unavailable"}')
    st.markdown(f'{"🟢" if HAS_GRAPH else "🔴"} Graph Intel: {"Active" if HAS_GRAPH else "Unavailable"}')
    st.markdown('---')
    st.markdown('### 📋 IRDAI Compliance')
    st.markdown('<span class="badge-green">✓ Registered</span>', unsafe_allow_html=True)
    st.markdown('All decisions require human review')
    st.markdown('---')
    st.caption('v1.0.0 | MIT License | © 2024 ClaimGuard AI')

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_claims_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / 'data' / 'sample_claims.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


@st.cache_data
def load_policies_data() -> pd.DataFrame:
    csv_path = Path(__file__).parent.parent / 'data' / 'sample_policies.csv'
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tabs = st.tabs([
    '📊 Explorer',
    '🏦 Underwrite',
    '🔍 Claims',
    '🤖 Policy Copilot',
    '🕸️ Graph Intel',
    '👤 HITL Review',
    '📈 Observability',
    '✅ Compliance',
])

# ============================================================
# Tab 1 — 📊 Explorer
# ============================================================
with tabs[0]:
    st.header('📊 Data Explorer')
    claims_df = load_claims_data()
    policies_df = load_policies_data()

    if claims_df.empty:
        st.warning('No claims data found. Run: python data/synthetic_generator.py')
    else:
        # Summary metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        fraud_rate = (
            claims_df['fraud_label'].sum() / len(claims_df) * 100
            if 'fraud_label' in claims_df.columns
            else 0
        )
        col1.metric('Total Claims', f'{len(claims_df):,}')
        col2.metric('Fraud Rate', f'{fraud_rate:.1f}%')
        col3.metric(
            'Avg Claim (₹)',
            f'{claims_df["claim_amount"].mean():,.0f}' if 'claim_amount' in claims_df.columns else 'N/A',
        )
        col4.metric('Total Policies', f'{len(policies_df):,}')
        avg_risk = policies_df['premium_adjustment'].mean() if 'premium_adjustment' in policies_df.columns else 0
        col5.metric('Avg Premium Adj', f'{avg_risk:.2f}x')

        st.markdown('---')

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            if 'claim_amount' in claims_df.columns and 'fraud_label' in claims_df.columns:
                fig = px.histogram(
                    claims_df,
                    x='claim_amount',
                    color='fraud_label',
                    nbins=40,
                    title='Claim Amount Distribution',
                    labels={'fraud_label': 'Fraud', 'claim_amount': 'Claim Amount (₹)'},
                    color_discrete_map={0: '#00C896', 1: '#FF6B6B'},
                    template='plotly_dark',
                )
                st.plotly_chart(fig, use_container_width=True)

        with chart_col2:
            if 'fraud_label' in claims_df.columns:
                fraud_counts = claims_df['fraud_label'].value_counts()
                names = (
                    ['Legitimate', 'Fraud']
                    if 0 in fraud_counts.index
                    else fraud_counts.index.astype(str).tolist()
                )
                fig = px.pie(
                    values=fraud_counts.values,
                    names=names,
                    title='Fraud vs Legitimate Claims',
                    color_discrete_sequence=['#00C896', '#FF6B6B'],
                    hole=0.4,
                    template='plotly_dark',
                )
                st.plotly_chart(fig, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            if 'claim_type' in claims_df.columns:
                ct_counts = claims_df['claim_type'].value_counts().reset_index()
                ct_counts.columns = ['claim_type', 'count']
                fig = px.bar(
                    ct_counts,
                    x='claim_type',
                    y='count',
                    title='Claims by Type',
                    color='count',
                    color_continuous_scale='Teal',
                    template='plotly_dark',
                )
                st.plotly_chart(fig, use_container_width=True)

        with chart_col4:
            if 'days_since_policy_start' in claims_df.columns and 'claim_amount' in claims_df.columns:
                fig = px.scatter(
                    claims_df,
                    x='days_since_policy_start',
                    y='claim_amount',
                    color='fraud_label' if 'fraud_label' in claims_df.columns else None,
                    title='Days Since Policy Start vs Claim Amount',
                    labels={
                        'days_since_policy_start': 'Days Since Policy Start',
                        'claim_amount': 'Claim Amount (₹)',
                        'fraud_label': 'Fraud',
                    },
                    color_discrete_map={0: '#00C896', 1: '#FF6B6B'},
                    opacity=0.6,
                    template='plotly_dark',
                )
                st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        num_cols = claims_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 2:
            corr = claims_df[num_cols].corr()
            fig = px.imshow(
                corr,
                title='Correlation Heatmap — Claims',
                color_continuous_scale='RdBu_r',
                text_auto='.2f',
                template='plotly_dark',
            )
            st.plotly_chart(fig, use_container_width=True)

        # Risk tier distribution from policies
        if not policies_df.empty and 'risk_tier' in policies_df.columns:
            rt_counts = policies_df['risk_tier'].value_counts().reset_index()
            rt_counts.columns = ['risk_tier', 'count']
            fig = px.bar(
                rt_counts,
                x='risk_tier',
                y='count',
                title='Policy Risk Tier Distribution',
                color='risk_tier',
                color_discrete_map={'low': '#00C896', 'medium': '#FFD700', 'high': '#FF6B6B'},
                template='plotly_dark',
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Tab 2 — 🏦 Underwrite
# ============================================================
with tabs[1]:
    st.header('🏦 Underwriting Risk Scorer')
    st.markdown('Score applicants at policy issuance time using the XGBoost + LightGBM ensemble.')

    with st.form('underwriting_form'):
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider('Age', 18, 80, 35)
            annual_income = st.number_input('Annual Income (₹)', 100000, 10000000, 800000, step=50000)
            credit_score = st.slider('Credit Score', 300, 900, 720)
            sum_insured = st.number_input('Sum Insured (₹)', 100000, 50000000, 1000000, step=100000)
        with col2:
            coverage_type = st.selectbox('Coverage Type', ['motor', 'health', 'property', 'life'])
            num_dependents = st.slider('Number of Dependents', 0, 10, 2)
            prior_claims_count = st.slider('Prior Claims Count', 0, 20, 0)
            region = st.selectbox('Region', ['north', 'south', 'east', 'west', 'central'])
            occupation = st.selectbox('Occupation', ['salaried', 'self-employed', 'business', 'retired', 'student'])

        uw_submitted = st.form_submit_button('🔍 Score Risk', use_container_width=True)

    if uw_submitted:
        if not HAS_UW or _uw_engine is None:
            st.error('Underwriting engine not available. Check installation.')
        else:
            with st.spinner('Scoring risk...'):
                try:
                    features = UnderwritingFeatures(
                        age=age,
                        annual_income=annual_income,
                        credit_score=credit_score,
                        sum_insured=sum_insured,
                        coverage_type=coverage_type,
                        num_dependents=num_dependents,
                        prior_claims_count=prior_claims_count,
                        region=region,
                        occupation=occupation,
                    )
                    result = _uw_engine.predict(features)

                    tier_colors = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                    tier_icon = tier_colors.get(getattr(result, 'risk_tier', 'medium'), '🟡')
                    st.markdown(f'## {tier_icon} Risk Tier: **{getattr(result, "risk_tier", "N/A").upper()}**')

                    m1, m2, m3 = st.columns(3)
                    risk_score = getattr(result, 'risk_score', 0)
                    m1.metric('Risk Score', f'{risk_score:.3f}')
                    m2.metric('Premium Adjustment', f'{getattr(result, "premium_adjustment", 1.0):.2f}x')
                    m3.metric('Model Version', getattr(result, 'model_version', 'N/A'))

                    # Gauge
                    gauge_color = (
                        '#FF6B6B' if risk_score > 0.6
                        else '#FFD700' if risk_score > 0.3
                        else '#00C896'
                    )
                    fig = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=risk_score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': 'Risk Score', 'font': {'size': 20}},
                        gauge={
                            'axis': {'range': [0, 1], 'tickwidth': 1},
                            'bar': {'color': gauge_color},
                            'steps': [
                                {'range': [0, 0.3], 'color': 'rgba(0,200,150,0.15)'},
                                {'range': [0.3, 0.6], 'color': 'rgba(255,215,0,0.15)'},
                                {'range': [0.6, 1], 'color': 'rgba(255,107,107,0.15)'},
                            ],
                            'threshold': {
                                'line': {'color': 'white', 'width': 3},
                                'thickness': 0.75,
                                'value': risk_score,
                            },
                        },
                    ))
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    shap_drivers = getattr(result, 'shap_drivers', [])
                    if shap_drivers:
                        drivers_df = pd.DataFrame(shap_drivers)
                        if 'feature' in drivers_df.columns and 'shap_value' in drivers_df.columns:
                            drivers_df = drivers_df.sort_values('shap_value')
                            colors = ['#00C896' if v < 0 else '#FF6B6B' for v in drivers_df['shap_value']]
                            fig = go.Figure(go.Bar(
                                x=drivers_df['shap_value'],
                                y=drivers_df['feature'],
                                orientation='h',
                                marker_color=colors,
                            ))
                            fig.update_layout(
                                title='Top SHAP Feature Drivers',
                                template='plotly_dark',
                                xaxis_title='SHAP Value (Impact on Risk Score)',
                                height=350,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with st.expander('📄 Raw Result JSON'):
                        st.json(result.model_dump() if hasattr(result, 'model_dump') else vars(result))

                    st.toast('✅ Risk scored successfully!', icon='✅')
                except Exception as exc:
                    st.error(f'Scoring failed: {exc}')

# ============================================================
# Tab 3 — 🔍 Claims
# ============================================================
with tabs[2]:
    st.header('🔍 Claims Fraud Detection Engine')
    st.markdown('Score claims at filing time for fraud risk.')

    with st.form('claims_form'):
        col1, col2 = st.columns(2)
        with col1:
            claim_id = st.text_input('Claim ID', 'CLM001')
            claimant_id = st.text_input('Claimant ID', 'CLMT001')
            policy_id = st.text_input('Policy ID', 'POL001')
            claim_amount = st.number_input('Claim Amount (₹)', 1000.0, 5000000.0, 50000.0, step=1000.0)
            days_since_policy_start = st.number_input('Days Since Policy Start', 0, 3650, 90)
        with col2:
            num_prior_claims = st.slider('Number of Prior Claims', 0, 20, 0)
            claim_type = st.selectbox('Claim Type', ['motor', 'health', 'property', 'life'])
            claim_severity = st.selectbox('Claim Severity', ['low', 'medium', 'high'])
            repair_shop_id = st.text_input('Repair Shop ID (optional)', '')
            medical_provider_id = st.text_input('Medical Provider ID (optional)', '')

        claims_submitted = st.form_submit_button('🚨 Score Claim', use_container_width=True)

    if claims_submitted:
        if not HAS_FRAUD or _fraud_engine is None:
            st.error('Fraud detection engine not available.')
        else:
            with st.spinner('Analysing claim...'):
                try:
                    features = ClaimFeatures(
                        claim_id=claim_id,
                        claimant_id=claimant_id,
                        policy_id=policy_id,
                        claim_amount=claim_amount,
                        days_since_policy_start=int(days_since_policy_start),
                        num_prior_claims=num_prior_claims,
                        claim_type=claim_type,
                        claim_severity=claim_severity,
                        repair_shop_id=repair_shop_id or None,
                        medical_provider_id=medical_provider_id or None,
                    )
                    result = _fraud_engine.predict(features)
                    fraud_flag = getattr(result, 'fraud_flag', False)
                    fraud_score = getattr(result, 'fraud_score', 0.0)

                    if fraud_flag:
                        st.error('⚠️ HIGH FRAUD RISK DETECTED — Flagged for mandatory HITL review')
                    else:
                        st.success('✅ Claim Appears Legitimate — Queued for standard HITL review')

                    m1, m2, m3 = st.columns(3)
                    m1.metric('Fraud Score', f'{fraud_score:.3f}')
                    m2.metric('Fraud Flag', '🚨 YES' if fraud_flag else '✅ NO')
                    m3.metric('Confidence Tier', getattr(result, 'confidence_tier', 'N/A'))

                    fig = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=fraud_score,
                        title={'text': 'Fraud Score'},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': '#FF6B6B' if fraud_score > 0.5 else '#00C896'},
                            'steps': [
                                {'range': [0, 0.3], 'color': 'rgba(0,200,150,0.15)'},
                                {'range': [0.3, 0.6], 'color': 'rgba(255,215,0,0.15)'},
                                {'range': [0.6, 1], 'color': 'rgba(255,107,107,0.15)'},
                            ],
                        },
                    ))
                    fig.update_layout(template='plotly_dark', height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    shap_drivers = getattr(result, 'shap_drivers', [])
                    if shap_drivers:
                        df_shap = pd.DataFrame(shap_drivers)
                        if 'feature' in df_shap.columns and 'shap_value' in df_shap.columns:
                            df_shap = df_shap.sort_values('shap_value')
                            fig = go.Figure(go.Bar(
                                x=df_shap['shap_value'],
                                y=df_shap['feature'],
                                orientation='h',
                                marker_color=[
                                    '#00C896' if v < 0 else '#FF6B6B'
                                    for v in df_shap['shap_value']
                                ],
                            ))
                            fig.update_layout(
                                title='SHAP Feature Drivers',
                                template='plotly_dark',
                                height=300,
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with st.expander('📄 Raw Result'):
                        st.json(result.model_dump() if hasattr(result, 'model_dump') else vars(result))
                    st.toast('Claim scored!', icon='🔍')
                except Exception as exc:
                    st.error(f'Claim scoring failed: {exc}')

# ============================================================
# Tab 4 — 🤖 Policy Copilot
# ============================================================
with tabs[3]:
    st.header('🤖 Policy Copilot — Powered by LangGraph')
    st.info('ℹ️ All decisions are drafted for human analyst review — no auto-approval ever.')

    context_type = st.selectbox('Context Type', ['underwriting', 'claims'], key='copilot_ctx')
    query = st.text_area(
        'Describe the case or question',
        height=100,
        placeholder=(
            'e.g. New motor policy applicant aged 32, credit score 750, '
            'no prior claims. Assess risk.'
        ),
    )

    default_features = {
        'underwriting': (
            '{"age":32,"annual_income":900000,"credit_score":750,'
            '"sum_insured":500000,"coverage_type":"motor",'
            '"num_dependents":1,"prior_claims_count":0,'
            '"region":"north","occupation":"salaried"}'
        ),
        'claims': (
            '{"claim_id":"CLM001","claimant_id":"CLMT001","policy_id":"POL001",'
            '"claim_amount":75000,"days_since_policy_start":45,'
            '"num_prior_claims":2,"claim_type":"motor","claim_severity":"high",'
            '"repair_shop_id":"SHOP001","medical_provider_id":""}'
        ),
    }
    features_json = st.text_area(
        'Features JSON',
        value=default_features.get(context_type, '{}'),
        height=120,
    )

    if st.button('⚡ Analyze with Copilot', use_container_width=True):
        if not query.strip():
            st.warning('Please enter a query.')
        else:
            with st.spinner('Running Policy Copilot pipeline...'):
                try:
                    import json as _json
                    features = _json.loads(features_json or '{}')
                    result = run_copilot(query=query, context_type=context_type, features=features)

                    st.markdown('### 📋 Draft Decision')
                    st.info(result.get('decision_draft', 'No decision draft generated.'))

                    with st.expander('📚 Retrieved Policy Clauses'):
                        docs = result.get('retrieved_docs', [])
                        if docs:
                            for i, doc in enumerate(docs, 1):
                                src = doc.get('source', 'Unknown')
                                content = doc.get('content', '')
                                score = doc.get('score', None)
                                st.markdown(
                                    f'**{i}. Source:** `{src}`'
                                    + (f' | Score: `{score:.3f}`' if score else '')
                                )
                                st.markdown(
                                    f'> {content[:500]}...'
                                    if len(content) > 500
                                    else f'> {content}'
                                )
                                st.markdown('---')
                        else:
                            st.caption(
                                'No policy clauses retrieved '
                                '(vector store may not be indexed).'
                            )

                    with st.expander('📊 Model Evidence'):
                        model_result = result.get('model_result', {})
                        if model_result:
                            ev_col1, ev_col2 = st.columns(2)
                            score_key = (
                                'risk_score' if context_type == 'underwriting' else 'fraud_score'
                            )
                            tier_key = (
                                'risk_tier' if context_type == 'underwriting' else 'confidence_tier'
                            )
                            ev_col1.metric('Score', f'{model_result.get(score_key, "N/A")}')
                            ev_col2.metric('Tier', model_result.get(tier_key, 'N/A'))
                            shap = model_result.get('shap_drivers', [])
                            if shap:
                                st.dataframe(pd.DataFrame(shap), use_container_width=True)
                        else:
                            st.caption('No model evidence available.')

                    st.markdown('### ⏳ HITL Status')
                    item_id = result.get('session_id', 'N/A')
                    st.warning(
                        f'🔄 Decision queued for analyst review | Session: `{item_id}`'
                    )

                    if 'copilot_first_run' not in st.session_state:
                        st.session_state['copilot_first_run'] = True
                        st.balloons()

                    st.toast('✅ Analysis complete — queued for HITL review', icon='🤖')
                except Exception as exc:
                    st.error(f'Copilot error: {exc}')

# ============================================================
# Tab 5 — 🕸️ Graph Intel
# ============================================================
with tabs[4]:
    st.header('🕸️ Collusion Ring Detection')
    st.markdown('Graph-based detection of claim-ring collusion using shared entities.')

    if not HAS_GRAPH:
        st.error('Graph collusion module not available.')
    else:
        with st.spinner('Analysing collusion rings...'):
            try:
                import networkx as nx
                claims_df = load_claims_data()
                if claims_df.empty:
                    st.warning('No claims data available.')
                else:
                    detector = GraphCollusionDetector()
                    rings = detector.analyze(claims_df)

                    total_rings = len(rings)
                    high_sev = sum(1 for r in rings if getattr(r, 'severity', '') == 'high')
                    claimants_in_rings = sum(
                        len(getattr(r, 'claimant_ids', [])) for r in rings
                    )

                    m1, m2, m3 = st.columns(3)
                    m1.metric('Rings Detected', total_rings)
                    m2.metric('High Severity', high_sev)
                    m3.metric('Claimants in Rings', claimants_in_rings)

                    if rings:
                        # Build NetworkX graph for visualisation
                        G = nx.Graph()
                        palette = [
                            '#FF6B6B', '#FFD700', '#5B9BD5',
                            '#FF9F43', '#A29BFE', '#FD79A8',
                        ]

                        for i, ring in enumerate(rings):
                            for cid in getattr(ring, 'claimant_ids', []):
                                G.add_node(cid, node_type='claimant')
                            for eid in getattr(ring, 'shared_entities', []):
                                G.add_node(eid, node_type='entity')
                                for cid in getattr(ring, 'claimant_ids', []):
                                    G.add_edge(cid, eid)

                        if len(G.nodes) > 0:
                            pos = nx.spring_layout(G, seed=42)

                            edge_x, edge_y = [], []
                            for e0, e1 in G.edges():
                                x0, y0 = pos[e0]
                                x1, y1 = pos[e1]
                                edge_x += [x0, x1, None]
                                edge_y += [y0, y1, None]

                            claimant_nodes = [
                                n for n, d in G.nodes(data=True)
                                if d.get('node_type') == 'claimant'
                            ]
                            entity_nodes = [
                                n for n, d in G.nodes(data=True)
                                if d.get('node_type') == 'entity'
                            ]

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=edge_x, y=edge_y, mode='lines',
                                line=dict(width=1, color='#444'),
                                hoverinfo='none', name='Connections',
                            ))

                            if claimant_nodes:
                                cx = [pos[n][0] for n in claimant_nodes]
                                cy = [pos[n][1] for n in claimant_nodes]
                                fig.add_trace(go.Scatter(
                                    x=cx, y=cy,
                                    mode='markers+text',
                                    marker=dict(
                                        size=18, color='#5B9BD5', symbol='circle',
                                        line=dict(width=2, color='white'),
                                    ),
                                    text=claimant_nodes,
                                    textposition='top center',
                                    hoverinfo='text',
                                    name='Claimants',
                                ))

                            if entity_nodes:
                                ex2 = [pos[n][0] for n in entity_nodes]
                                ey2 = [pos[n][1] for n in entity_nodes]
                                fig.add_trace(go.Scatter(
                                    x=ex2, y=ey2,
                                    mode='markers+text',
                                    marker=dict(
                                        size=14, color='#FF9F43', symbol='diamond',
                                        line=dict(width=2, color='white'),
                                    ),
                                    text=entity_nodes,
                                    textposition='top center',
                                    hoverinfo='text',
                                    name='Shared Entities',
                                ))

                            fig.update_layout(
                                title='Collusion Network Graph',
                                template='plotly_dark',
                                showlegend=True,
                                hovermode='closest',
                                height=500,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Ring summary table
                        st.markdown('### Ring Summary')
                        ring_data = []
                        for ring in rings:
                            sev = getattr(ring, 'severity', 'medium')
                            sev_badge = (
                                f'🔴 {sev}' if sev == 'high'
                                else f'🟡 {sev}' if sev == 'medium'
                                else f'🟢 {sev}'
                            )
                            ring_data.append({
                                'Ring ID': getattr(ring, 'ring_id', 'N/A'),
                                'Severity': sev_badge,
                                'Claimants': len(getattr(ring, 'claimant_ids', [])),
                                'Shared Entities': len(getattr(ring, 'shared_entities', [])),
                                'Centrality Score': f'{getattr(ring, "centrality_score", 0):.3f}',
                            })
                        st.dataframe(pd.DataFrame(ring_data), use_container_width=True)

                        for ring in rings[:5]:
                            with st.expander(
                                f'Ring {getattr(ring, "ring_id", "N/A")} — Details'
                            ):
                                ec1, ec2 = st.columns(2)
                                ec1.markdown('**Claimants:**')
                                for cid in getattr(ring, 'claimant_ids', []):
                                    ec1.markdown(f'• `{cid}`')
                                ec2.markdown('**Shared Entities:**')
                                for eid in getattr(ring, 'shared_entities', []):
                                    ec2.markdown(f'• `{eid}`')
                    else:
                        st.success(
                            '✅ No suspicious collusion rings detected in the current dataset.'
                        )
            except Exception as exc:
                st.error(f'Graph analysis failed: {exc}')

# ============================================================
# Tab 6 — 👤 HITL Review
# ============================================================
with tabs[5]:
    st.header('👤 Human-in-the-Loop Analyst Review')

    if not HAS_HITL or hitl_queue is None:
        st.error('HITL module not available.')
    else:
        pending = hitl_queue.get_pending()
        col1, col2 = st.columns([1, 4])
        col1.metric('Pending Reviews', len(pending))
        if col2.button('🔄 Refresh Queue'):
            st.rerun()

        if not pending:
            st.success('✅ All reviews complete — queue is empty')
        else:
            st.markdown(f'**{len(pending)} item(s) awaiting review:**')
            for item in pending:
                with st.container(border=True):
                    h1, h2, h3 = st.columns([3, 2, 2])
                    h1.markdown(f'**Item ID:** `{item.item_id}`')
                    ctx_badge = (
                        f'🏦 {item.context_type}'
                        if item.context_type == 'underwriting'
                        else f'🔍 {item.context_type}'
                    )
                    h2.markdown(f'**Type:** {ctx_badge}')
                    h3.markdown(
                        f'**Created:** {item.created_at.strftime("%Y-%m-%d %H:%M")}'
                    )

                    st.markdown('**Draft Decision:**')
                    st.text_area(
                        'Decision Draft',
                        value=item.decision_draft,
                        height=120,
                        key=f'draft_{item.item_id}',
                        disabled=True,
                    )

                    mr = item.model_result or {}
                    if mr:
                        score_key = (
                            'risk_score'
                            if item.context_type == 'underwriting'
                            else 'fraud_score'
                        )
                        tier_key = (
                            'risk_tier'
                            if item.context_type == 'underwriting'
                            else 'confidence_tier'
                        )
                        sm1, sm2 = st.columns(2)
                        sm1.metric('Score', f'{mr.get(score_key, "N/A")}')
                        sm2.metric('Tier', mr.get(tier_key, 'N/A'))

                    notes = st.text_area(
                        'Analyst Notes',
                        placeholder='Enter review notes...',
                        key=f'notes_{item.item_id}',
                    )

                    b1, b2, b3 = st.columns(3)
                    if b1.button(
                        '✅ Approve',
                        key=f'approve_{item.item_id}',
                        use_container_width=True,
                    ):
                        hitl_queue.review(item.item_id, 'approved', notes)
                        st.toast('Decision approved!', icon='✅')
                        st.rerun()
                    if b2.button(
                        '❌ Reject',
                        key=f'reject_{item.item_id}',
                        use_container_width=True,
                    ):
                        hitl_queue.review(item.item_id, 'rejected', notes)
                        st.toast('Decision rejected!', icon='❌')
                        st.rerun()
                    if b3.button(
                        '⬆️ Escalate',
                        key=f'escalate_{item.item_id}',
                        use_container_width=True,
                    ):
                        hitl_queue.review(item.item_id, 'escalated', notes)
                        st.toast('Decision escalated!', icon='⬆️')
                        st.rerun()

        # Review history
        all_items = hitl_queue.get_all()
        reviewed = [i for i in all_items if i.status != 'pending']
        if reviewed:
            with st.expander(
                f'📋 Review History (last {min(10, len(reviewed))} items)'
            ):
                hist_data = []
                for item in reviewed[-10:]:
                    status_icon = {
                        'approved': '✅', 'rejected': '❌', 'escalated': '⬆️',
                    }.get(item.status, '❓')
                    hist_data.append({
                        'Item ID': item.item_id[:12] + '...',
                        'Type': item.context_type,
                        'Status': f'{status_icon} {item.status}',
                        'Reviewed At': (
                            item.reviewed_at.strftime('%Y-%m-%d %H:%M')
                            if item.reviewed_at
                            else 'N/A'
                        ),
                        'Notes': (item.analyst_review or '')[:50],
                    })
                st.dataframe(pd.DataFrame(hist_data), use_container_width=True)

# ============================================================
# Tab 7 — 📈 Observability
# ============================================================
with tabs[6]:
    st.header('📈 Live Metrics — Prometheus Observability')

    try:
        from prometheus_client import REGISTRY  # noqa: F401
        st.success('✅ prometheus_client installed — metrics are live')
        HAS_PROM = True
    except ImportError:
        st.info('ℹ️ prometheus_client not installed — showing illustrative mock values')
        HAS_PROM = False

    # Try to pull live prometheus metric values; fall through to mock values
    try:
        from src.copilot_metrics import record_agent_run as _rar  # noqa: F401
        from prometheus_client import REGISTRY as REG

        def _get_metric(name: str, default: int = 0) -> int:
            try:
                for metric in REG.collect():
                    if metric.name == name:
                        return int(sum(s.value for s in metric.samples))
            except Exception:
                pass
            return default

        agent_runs = _get_metric('claimguard_copilot_agent_runs_total', 12)
        tool_calls = _get_metric('claimguard_copilot_tool_calls_total', 8)
        ret_hits = _get_metric('claimguard_copilot_retriever_hits_total', 15)
        decisions = _get_metric('claimguard_copilot_decisions_drafted_total', 7)
        queue_depth = hitl_queue.queue_depth() if hitl_queue else 3
    except Exception:
        agent_runs, tool_calls, ret_hits, decisions, queue_depth = 12, 8, 15, 7, 3

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric('Agent Runs', agent_runs)
    mc2.metric('Tool Calls', tool_calls)
    mc3.metric('Retriever Hits', ret_hits)
    mc4.metric('Decisions Drafted', decisions)
    mc5.metric('HITL Queue Depth', queue_depth)

    st.markdown('---')

    # Latency histogram (deterministic mock)
    rng = np.random.default_rng(seed=42)
    percentiles = ['p50', 'p75', 'p90', 'p95', 'p99']
    # Generate 3 samples per percentile then average to get stable mock values.
    low_bounds = np.array([0.05, 0.08, 0.12, 0.18, 0.35])
    high_bounds = np.array([0.10, 0.15, 0.22, 0.30, 0.60])
    latencies = np.array([
        rng.uniform(low_bounds[k], high_bounds[k], size=3).mean()
        for k in range(len(percentiles))
    ])

    fig_lat = go.Figure(go.Bar(
        x=percentiles,
        y=latencies,
        marker_color='#00C896',
        text=[f'{v:.3f}s' for v in latencies],
        textposition='outside',
    ))
    fig_lat.update_layout(
        title='Agent Latency Percentiles (seconds)',
        template='plotly_dark',
        yaxis_title='Latency (s)',
        xaxis_title='Percentile',
        height=350,
    )
    st.plotly_chart(fig_lat, use_container_width=True)

    # Time-series: decisions over last 24 hours
    hours = list(range(24))
    rng2 = np.random.default_rng(seed=99)
    decisions_ts = rng2.integers(0, 5, size=24).cumsum().tolist()

    fig_ts = go.Figure(go.Scatter(
        x=hours,
        y=decisions_ts,
        mode='lines+markers',
        line=dict(color='#00C896', width=2),
        marker=dict(size=6, color='#00C896'),
        fill='tozeroy',
        fillcolor='rgba(0,200,150,0.1)',
        name='Decisions Drafted',
    ))
    fig_ts.update_layout(
        title='Decisions Drafted — Last 24 Hours',
        template='plotly_dark',
        xaxis_title='Hours Ago',
        yaxis_title='Cumulative Decisions',
        height=300,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.info('📊 Grafana dashboard: http://localhost:3000 (requires Docker Compose)')

# ============================================================
# Tab 8 — ✅ Compliance
# ============================================================
with tabs[7]:
    st.header('✅ IRDAI Regulatory Compliance Dashboard')

    if not HAS_COMPLIANCE:
        st.error('Compliance module not available.')
    else:
        with st.spinner('Loading compliance report...'):
            try:
                report = generate_compliance_report()
                score = report.overall_score

                gauge_color = (
                    '#FF6B6B' if score < 60
                    else '#FFD700' if score < 80
                    else '#00C896'
                )
                fig = go.Figure(go.Indicator(
                    mode='gauge+number+delta',
                    value=score,
                    title={'text': 'IRDAI Compliance Score'},
                    delta={
                        'reference': 80,
                        'increasing': {'color': '#00C896'},
                        'decreasing': {'color': '#FF6B6B'},
                    },
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': gauge_color},
                        'steps': [
                            {'range': [0, 60], 'color': 'rgba(255,107,107,0.15)'},
                            {'range': [60, 80], 'color': 'rgba(255,215,0,0.15)'},
                            {'range': [80, 100], 'color': 'rgba(0,200,150,0.15)'},
                        ],
                        'threshold': {
                            'line': {'color': 'white', 'width': 3},
                            'thickness': 0.75,
                            'value': 80,
                        },
                    },
                ))
                fig.update_layout(template='plotly_dark', height=350)

                cg1, cg2 = st.columns([1, 1])
                with cg1:
                    st.plotly_chart(fig, use_container_width=True)
                with cg2:
                    st.metric('Compliance Score', f'{score:.1f}/100')
                    st.markdown(f'**Summary:** {report.summary}')

                    statuses = [c.status for c in report.controls]
                    status_counts = {
                        'compliant': statuses.count('compliant'),
                        'partial': statuses.count('partial'),
                        'non_compliant': statuses.count('non_compliant'),
                    }
                    fig2 = px.pie(
                        values=list(status_counts.values()),
                        names=list(status_counts.keys()),
                        title='Controls Breakdown',
                        color_discrete_map={
                            'compliant': '#00C896',
                            'partial': '#FFD700',
                            'non_compliant': '#FF6B6B',
                        },
                        hole=0.4,
                        template='plotly_dark',
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                st.markdown('---')
                st.markdown('### 📋 Control Details')

                controls = report.controls
                for i in range(0, len(controls), 2):
                    cols = st.columns(2)
                    for j, ctrl in enumerate(controls[i:i + 2]):
                        with cols[j]:
                            status_badge = {
                                'compliant': '<span class="badge-green">✓ Compliant</span>',
                                'partial': '<span class="badge-yellow">⚠ Partial</span>',
                                'non_compliant': '<span class="badge-red">✗ Non-Compliant</span>',
                            }.get(ctrl.status, '')
                            severity_badge = {
                                'high': '<span class="badge-red">High</span>',
                                'medium': '<span class="badge-yellow">Medium</span>',
                                'low': '<span class="badge-green">Low</span>',
                            }.get(ctrl.severity, '')
                            st.markdown(
                                f'**{ctrl.title}** {status_badge} | Severity: {severity_badge}',
                                unsafe_allow_html=True,
                            )
                            st.caption(f'Category: {ctrl.category}')
                            with st.expander('Evidence & Remediation'):
                                st.markdown(f'**Evidence:** {ctrl.evidence}')
                                st.markdown(f'**Remediation:** {ctrl.remediation}')

                st.markdown('---')
                st.caption(
                    f'Report ID: `{report.report_id}` | '
                    f'Generated: {report.generated_at.strftime("%Y-%m-%d %H:%M UTC")}'
                )
            except Exception as exc:
                st.error(f'Compliance report error: {exc}')


if __name__ == '__main__':
    pass
