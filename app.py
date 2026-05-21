"""
dashboard/app.py
AI Grid Resilience Copilot — Streamlit Dashboard

Run:
    cd C:\\Users\\Zishan\\ai-grid-resilience-copilot
    .venv\\Scripts\\activate
    pip install plotly pillow
    streamlit run dashboard\\app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    st.error("Plotly is not installed. Run: pip install plotly")
    st.stop()

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================
# PATHS
# ============================================================
ROOT = Path(__file__).resolve().parents[1]

PHASE8 = ROOT / "results" / "phase8_final"
PHASE4 = ROOT / "results" / "phase4"
PHASE6 = ROOT / "results" / "phase6"
PHASE10 = ROOT / "results" / "phase10_cascading"
PHASE11 = ROOT / "results" / "phase11"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AI Grid Resilience Copilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
}

.hero-panel {
    background: linear-gradient(135deg, #0f172a 0%, #0f4c81 55%, #1e88a8 100%);
    color: white;
    border-radius: 18px;
    padding: 1.35rem 1.55rem;
    margin-bottom: 1.0rem;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.18);
}

.hero-panel h1 {
    color: white;
    font-size: 2.0rem;
    margin-bottom: 0.3rem;
}

.hero-panel p {
    color: rgba(255,255,255,0.88);
    font-size: 0.98rem;
    margin-bottom: 0;
}

.metric-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 0.9rem 1rem;
    box-shadow: 0 6px 20px rgba(15,23,42,0.06);
    min-height: 95px;
}

.metric-card .label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.metric-card .value {
    color: #0f4c81;
    font-size: 1.45rem;
    font-weight: 800;
    margin-top: 0.2rem;
}

.metric-card .sub {
    color: #64748b;
    font-size: 0.76rem;
    margin-top: 0.12rem;
}

.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 0.95rem 1.1rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 20px rgba(15,23,42,0.04);
}

.badge {
    display: inline-block;
    background: #e0f2fe;
    color: #075985;
    border: 1px solid #bae6fd;
    border-radius: 999px;
    padding: 0.18rem 0.65rem;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 0.35rem;
}

.small-muted {
    color: #64748b;
    font-size: 0.86rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def load_csv(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Could not read {path.name}: {e}")
            return None
    return None


def fmt(x, digits: int = 3):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}"
    except Exception:
        return "N/A"


def fmt_pct(x, digits: int = 1):
    try:
        if x is None or pd.isna(x):
            return "N/A"
        return f"{float(x):.{digits}f}%"
    except Exception:
        return "N/A"


def metric_card(label: str, value: str, sub: str = ""):
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """


def section_card(html: str):
    st.markdown(f'<div class="section-card">{html}</div>', unsafe_allow_html=True)


def missing(msg: str):
    st.warning(msg)


def to_num(series):
    return pd.to_numeric(series, errors="coerce")


def safe_plotly(fig):
    # New Streamlit supports width="stretch". Older versions use use_container_width.
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def safe_df(df, height="content"):
    # Your Streamlit version crashes if height=None, so never pass None.
    if height is None:
        height = "content"
    try:
        st.dataframe(df, width="stretch", height=height)
    except TypeError:
        if height == "content":
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True, height=height)


def safe_image(img, caption=None):
    # Avoid deprecated use_container_width warning where possible.
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


MODEL_LABELS = {
    "mlp": "MLP",
    "gcn": "GCN",
    "gat": "GAT",
    "stgcn": "STGCN",
    "stgat_nogate": "STGAT-NoGate (Gate Ablation)",
    "STGAT-NoGate": "STGAT-NoGate (Gate Ablation)",
    "stgat-nogate": "STGAT-NoGate (Gate Ablation)",
    "resiligraph_stgat": "ResiliGraph-STGAT",
    "Res.-STGAT": "ResiliGraph-STGAT",
    "res.-stgat": "ResiliGraph-STGAT",
}

MODEL_ORDER = [
    "MLP",
    "GCN",
    "GAT",
    "STGCN",
    "STGAT-NoGate (Gate Ablation)",
    "ResiliGraph-STGAT",
]

MODEL_COLORS = {
    "MLP": "#94a3b8",
    "GCN": "#64748b",
    "GAT": "#3b82f6",
    "STGCN": "#10b981",
    "STGAT-NoGate (Gate Ablation)": "#f59e0b",
    "ResiliGraph-STGAT": "#ef4444",
}


def clean_model_name(x):
    s = str(x).strip()
    low = s.lower()
    if low in MODEL_LABELS:
        return MODEL_LABELS[low]
    if s in MODEL_LABELS:
        return MODEL_LABELS[s]
    return s


def model_sort_key(x):
    try:
        return MODEL_ORDER.index(clean_model_name(x))
    except ValueError:
        return 99


def find_col(df, candidates):
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None


def bar_fig(df, x, y, title, y_range=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=x,
        color_discrete_map=MODEL_COLORS,
        text=y,
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=55, b=20, l=10, r=10),
        xaxis_title="",
        yaxis_title="",
    )
    if y_range is not None:
        fig.update_yaxes(range=y_range)
    return fig


def risk_badge(risk: str):
    risk = str(risk).lower()
    if risk == "fault":
        return "🔴 Fault Precursor"
    if risk == "congestion":
        return "🟠 Congestion Precursor"
    if risk == "voltage":
        return "🔵 Voltage Precursor"
    return "🟢 Normal"


def get_cascade_reduction(summary):
    if summary is None or "Metric" not in summary.columns or "Value" not in summary.columns:
        return None
    try:
        mask = summary["Metric"].astype(str).str.contains("reduction", case=False, na=False)
        if mask.any():
            return float(summary.loc[mask, "Value"].iloc[0])
    except Exception:
        return None
    return None


# ============================================================
# LOAD DATA
# ============================================================
table1 = load_csv(PHASE8 / "table1_model_comparison.csv")
table2 = load_csv(PHASE8 / "table2_renewable_comparison.csv")
table3 = load_csv(PHASE8 / "table3_copilot_examples.csv")
table4 = load_csv(PHASE8 / "table4_rl_performance.csv")
table5 = load_csv(PHASE8 / "table5_explainability_critical_lines.csv")
table6 = load_csv(PHASE8 / "table6_rl_action_distribution.csv")

attn_corr = load_csv(PHASE4 / "attention_loading_correlation.csv")
top_crit = load_csv(PHASE4 / "top_critical_lines.csv")
top_risk_b = load_csv(PHASE4 / "top_risk_buses.csv")

full_copilot = load_csv(PHASE6 / "copilot_recommendations_full.csv")
if table3 is None:
    table3 = load_csv(PHASE6 / "copilot_recommendations.csv")
if full_copilot is None:
    full_copilot = table3

cas_summary = load_csv(PHASE10 / "cascading_summary.csv")
cas_series = load_csv(PHASE10 / "cascading_risk_timeseries.csv")

stress_summary = load_csv(PHASE11 / "stress_benchmark_summary.csv")
stress_best = load_csv(PHASE11 / "stress_best_by_level.csv")
stress_meta = load_csv(PHASE11 / "stress_dataset_metadata.csv")

if table5 is None:
    table5 = top_crit


# ============================================================
# PREPARE COMMON DATA
# ============================================================
if table1 is not None and len(table1) > 0:
    table1 = table1.copy()
    if "Model" in table1.columns:
        table1["Model Clean"] = table1["Model"].apply(clean_model_name)
    elif "model" in table1.columns:
        table1["Model Clean"] = table1["model"].apply(clean_model_name)
    else:
        table1["Model Clean"] = table1.index.astype(str)

    table1["_order"] = table1["Model Clean"].apply(model_sort_key)
    table1 = table1.sort_values("_order").reset_index(drop=True)

f1_col = find_col(table1, ["Mean F1 (mean)", "macro_f1_mean", "Mean F1", "macro_f1"])
auc_col = find_col(table1, ["Mean AUC (mean)", "macro_auc_mean", "Mean AUC", "macro_auc"])
auprc_col = find_col(table1, ["Mean AUPRC (mean)", "macro_auprc_mean", "Mean AUPRC", "macro_auprc"])
recall_col = find_col(table1, ["Mean Recall (mean)", "macro_recall_mean", "Mean Recall", "macro_recall"])

best_balanced_model = "N/A"
best_f1 = None
best_recall_model = "N/A"
best_recall = None
resili_recall = None

if table1 is not None and f1_col:
    tmp = table1.copy()
    tmp["_f1"] = to_num(tmp[f1_col])
    row = tmp.sort_values("_f1", ascending=False).iloc[0]
    best_balanced_model = row["Model Clean"]
    best_f1 = row["_f1"]

if table1 is not None and recall_col:
    tmp = table1.copy()
    tmp["_recall"] = to_num(tmp[recall_col])
    row = tmp.sort_values("_recall", ascending=False).iloc[0]
    best_recall_model = row["Model Clean"]
    best_recall = row["_recall"]

    res_rows = tmp[tmp["Model Clean"] == "ResiliGraph-STGAT"]
    if len(res_rows) > 0:
        resili_recall = float(res_rows["_recall"].iloc[0])

renew_gain = None
voltage_gain = None
if table2 is not None and len(table2) >= 2:
    t2 = table2.copy()
    if "Configuration" in t2.columns:
        base_rows = t2[t2["Configuration"].astype(str).str.contains("Without|w/o", case=False, na=False)]
        der_rows = t2[t2["Configuration"].astype(str).str.contains("With|solar|DER", case=False, na=False)]
        base = base_rows.iloc[0] if len(base_rows) else t2.iloc[0]
        der = der_rows.iloc[-1] if len(der_rows) else t2.iloc[1]
    else:
        base = t2.iloc[0]
        der = t2.iloc[1]

    if "Mean F1" in t2.columns:
        renew_gain = float(der["Mean F1"] - base["Mean F1"])
    if "Voltage F1" in t2.columns:
        voltage_gain = float(der["Voltage F1"] - base["Voltage F1"])

total_cases = risky_cases = normal_cases = None
dominant_risk = "N/A"
risk_summary = None

if full_copilot is not None and len(full_copilot) > 0 and "risk_type" in full_copilot.columns:
    total_cases = len(full_copilot)
    normal_cases = len(full_copilot[full_copilot["risk_type"] == "normal"])
    risky_cases = total_cases - normal_cases
    risky_only = full_copilot[full_copilot["risk_type"] != "normal"]
    dominant_risk = (
        risky_only["risk_type"].value_counts().idxmax().title()
        if len(risky_only) > 0
        else "Normal"
    )
    risk_summary = full_copilot["risk_type"].value_counts().reset_index()
    risk_summary.columns = ["Risk Type", "Cases"]

cascade_reduction = get_cascade_reduction(cas_summary)


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div class="hero-panel">
    <h1>⚡ AI Grid Resilience Copilot</h1>
    <p>Predictive fault, congestion, and voltage-instability precursor intelligence for renewable-rich distribution feeders.</p>
</div>
""",
    unsafe_allow_html=True,
)

h1, h2, h3, h4, h5 = st.columns(5)
h1.markdown(metric_card("Best Balanced Model", best_balanced_model, "Macro F1"), unsafe_allow_html=True)
h2.markdown(metric_card("Best Safety Model", best_recall_model, "Macro Recall"), unsafe_allow_html=True)
h3.markdown(metric_card("ResiliGraph Recall", fmt(resili_recall), "Safety-critical metric"), unsafe_allow_html=True)
h4.markdown(metric_card("Voltage F1 Gain", f"+{fmt(voltage_gain)}", "DER features"), unsafe_allow_html=True)
h5.markdown(metric_card("CPU Inference", "31.30 ms", "T = 20 sequence"), unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    [
        "🖥️ Control Center",
        "📊 Model Performance",
        "🌞 Renewable Intelligence",
        "🔎 Explainability",
        "🤖 AI Copilot Actions",
        "🛡️ RL Mitigation",
        "🌊 Cascading Risk",
        "📈 Stress Benchmark",
        "📸 LinkedIn Snapshot",
    ]
)


# ============================================================
# TAB 1 — CONTROL CENTER
# ============================================================
with tab1:
    st.subheader("🖥️ Executive Control Center")

    section_card(
        "<b>System narrative:</b> Temporal graph learning enables early-warning prediction "
        "of fault, congestion, and voltage-instability precursors in renewable-rich distribution feeders."
    )

    c1, c2, c3, c4 = st.columns(4)
    stability = 100.0 * normal_cases / total_cases if total_cases else None
    c1.markdown(metric_card("Grid Stability Index", fmt_pct(stability), "normal bus-time cases"), unsafe_allow_html=True)
    c2.markdown(metric_card("Dominant Threat", dominant_risk, "from copilot output"), unsafe_allow_html=True)
    c3.markdown(metric_card("Risky Cases", f"{risky_cases:,}" if risky_cases is not None else "N/A", "bus-time cases"), unsafe_allow_html=True)
    c4.markdown(metric_card("RL Improvement", "+7.8%", "reward over rule-based"), unsafe_allow_html=True)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Critical Feeder Attention Map")
        fig_path = PHASE4 / "attention_feeder.png"
        if fig_path.exists() and Image is not None:
            safe_image(Image.open(fig_path), caption="Model-influence attention over IEEE 33-bus feeder.")
        else:
            missing("Attention map missing. Run: python scripts/run_phase4_explainability.py")

    with right:
        st.markdown("#### Top Operator Recommendations")
        if table3 is not None and len(table3) > 0:
            for _, row in table3.head(3).iterrows():
                st.markdown(f"**{risk_badge(row.get('risk_type', 'normal'))} — Bus {row.get('bus', 'N/A')}**")
                p1, p2, p3 = st.columns(3)
                p1.metric("Fault", fmt(row.get("fault_prob", 0)))
                p2.metric("Cong.", fmt(row.get("congestion_prob", 0)))
                p3.metric("Voltage", fmt(row.get("voltage_prob", 0)))
                st.caption(f"Action: `{row.get('recommended_action', 'monitor')}`")
                st.divider()
        else:
            missing("Copilot output missing. Run: python scripts/run_phase6_copilot.py")

    if risk_summary is not None:
        st.markdown("#### Risk Distribution Across Test Set")
        fig_r = px.bar(
            risk_summary,
            x="Risk Type",
            y="Cases",
            color="Risk Type",
            color_discrete_map={
                "fault": "#ef4444",
                "congestion": "#f59e0b",
                "voltage": "#3b82f6",
                "normal": "#10b981",
            },
            title="Bus-Timestep Risk Distribution",
            text="Cases",
        )
        fig_r.update_traces(textposition="outside")
        fig_r.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
        safe_plotly(fig_r)


# ============================================================
# TAB 2 — MODEL PERFORMANCE
# ============================================================
with tab2:
    st.subheader("📊 Model Performance Benchmark")

    section_card(
        "<b>Interpretation:</b> ResiliGraph-STGAT is not claimed as best on every metric. "
        "It is competitive with STGCN on AUPRC while adding explainability, renewable-stress conditioning, "
        "and operator recommendation capability. STGCN remains the strongest balanced performer on some metrics; "
        "ResiliGraph-STGAT achieves the highest Recall."
    )

    if table1 is not None and len(table1) > 0:
        view_cols = ["Model Clean"]
        for c in [f1_col, auc_col, auprc_col, recall_col]:
            if c and c not in view_cols:
                view_cols.append(c)

        safe_df(table1[view_cols].rename(columns={"Model Clean": "Model"}))

        if auprc_col:
            st.markdown("#### Macro AUPRC — Primary Paper Metric")
            t = table1.copy()
            t[auprc_col] = to_num(t[auprc_col])
            safe_plotly(bar_fig(t, "Model Clean", auprc_col, "Macro AUPRC by Model", [0.75, 1.02]))

        if recall_col:
            st.markdown("#### Macro Recall — Safety-Critical Metric")
            t = table1.copy()
            t[recall_col] = to_num(t[recall_col])
            safe_plotly(bar_fig(t, "Model Clean", recall_col, "Macro Recall by Model", [0.75, 1.02]))

        col_a, col_b = st.columns(2)
        with col_a:
            if f1_col:
                t = table1.copy()
                t[f1_col] = to_num(t[f1_col])
                safe_plotly(bar_fig(t, "Model Clean", f1_col, "Macro F1 by Model", [0.75, 1.02]))

        with col_b:
            if auc_col:
                t = table1.copy()
                t[auc_col] = to_num(t[auc_col])
                safe_plotly(bar_fig(t, "Model Clean", auc_col, "Macro AUC by Model", [0.90, 1.01]))

        with st.expander("Computational Efficiency"):
            eff = pd.DataFrame(
                [
                    {"Model": "MLP", "Parameters": "5.0K", "Avg Inference": "0.08 ms", "Complexity": "Low"},
                    {"Model": "GCN", "Parameters": "5.0K", "Avg Inference": "0.73 ms", "Complexity": "Low"},
                    {"Model": "GAT", "Parameters": "19.8K", "Avg Inference": "1.13 ms", "Complexity": "Moderate"},
                    {"Model": "STGCN", "Parameters": "30.0K", "Avg Inference": "15.27 ms", "Complexity": "Moderate"},
                    {"Model": "ResiliGraph-STGAT", "Parameters": "97.4K", "Avg Inference": "31.30 ms", "Complexity": "Moderate-High"},
                ]
            )
            safe_df(eff)
            st.caption("Inference latency measured on CPU using a 20-step temporal graph sequence averaged across 50 forward passes.")

    else:
        missing("Model comparison missing. Run: python scripts/run_phase8_final_results.py")


# ============================================================
# TAB 3 — RENEWABLE INTELLIGENCE
# ============================================================
with tab3:
    st.subheader("🌞 Renewable-Aware Intelligence")

    if table2 is not None and len(table2) > 0:
        section_card(
            "<b>Finding:</b> Voltage precursor detection is highly sensitive to renewable-aware features because "
            "DER variability affects feeder voltage behavior before threshold violations."
        )

        t = table2.copy()
        if "Configuration" in t.columns:
            t["Config"] = t["Configuration"].apply(
                lambda s: "Without DER Channels" if "without" in str(s).lower() or "w/o" in str(s).lower() else "With DER Channels"
            )
        else:
            t["Config"] = ["Without DER Channels", "With DER Channels"][: len(t)]

        metric_cols = [c for c in ["Fault F1", "Congestion F1", "Voltage F1", "Mean AUPRC"] if c in t.columns]
        plot_df = t.melt(id_vars="Config", value_vars=metric_cols, var_name="Metric", value_name="Score")
        fig = px.bar(plot_df, x="Metric", y="Score", color="Config", barmode="group", text="Score", title="Renewable Feature Ablation")
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", yaxis=dict(range=[0, 1.05]))
        safe_plotly(fig)

        safe_df(t)
    else:
        missing("Renewable comparison missing. Run: python scripts/run_phase8_final_results.py")


# ============================================================
# TAB 4 — EXPLAINABILITY
# ============================================================
with tab4:
    st.subheader("🔎 Attention-Based Explainability")

    section_card(
        "<b>Important:</b> Attention is model-influence attribution, not causal physical explanation. "
        "It shows which feeder edges received high model weighting during prediction."
    )

    fig_path = PHASE4 / "attention_feeder.png"
    if fig_path.exists() and Image is not None:
        safe_image(Image.open(fig_path), caption="Aggregated attention over IEEE 33-bus feeder.")
    else:
        missing("Attention map missing. Run: python scripts/run_phase4_explainability.py")

    if table5 is not None and len(table5) > 0:
        t = table5.copy()

        attn_col = find_col(t, ["attention_score", "attn_mean", "Attention", "Attn. Mean"])
        loading_col = find_col(t, ["loading_pct_mean", "Loading (%)", "loading_pct", "loading"])
        from_col = find_col(t, ["from_bus_name", "from_bus", "From Bus"])
        to_col = find_col(t, ["to_bus_name", "to_bus", "To Bus"])

        if from_col and to_col:
            t["Edge"] = t[from_col].astype(str) + " → " + t[to_col].astype(str)

        st.markdown("#### Top Critical Feeder Lines")
        show_cols = [c for c in ["rank", "Edge", from_col, to_col, attn_col, loading_col] if c and c in t.columns]
        safe_df(t[show_cols].head(10))

        if attn_col:
            t[attn_col] = to_num(t[attn_col])
            fig = px.bar(
                t.head(10),
                x="Edge" if "Edge" in t.columns else t.index,
                y=attn_col,
                title="Top 10 Edges by Mean Attention",
                text=attn_col,
            )
            fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            safe_plotly(fig)
    else:
        missing("Critical line table missing. Run: python scripts/run_phase4_explainability.py")

    if attn_corr is not None:
        st.markdown("#### Attention–Loading Correlation")
        safe_df(attn_corr)
        st.caption("Layer 1: rho = -0.096, p = 0.45. Layer 2: rho = -0.111, p = 0.38.")


# ============================================================
# TAB 5 — AI COPILOT ACTIONS
# ============================================================
with tab5:
    st.subheader("🤖 AI Copilot Actions")

    if full_copilot is not None and len(full_copilot) > 0:
        df = full_copilot.copy()

        risk_values = ["all"]
        if "risk_type" in df.columns:
            risk_values += sorted(df["risk_type"].dropna().astype(str).unique().tolist())

        risk_filter = st.selectbox("Risk type", risk_values)
        display = df.copy()
        if risk_filter != "all" and "risk_type" in display.columns:
            display = display[display["risk_type"].astype(str) == risk_filter]

        bus_col = find_col(display, ["bus", "Bus"])
        if bus_col:
            buses = ["all"] + sorted(display[bus_col].dropna().astype(str).unique().tolist())
            bus_filter = st.selectbox("Bus", buses)
            if bus_filter != "all":
                display = display[display[bus_col].astype(str) == bus_filter]

        for _, row in display.head(8).iterrows():
            st.markdown(f"### {risk_badge(row.get('risk_type', 'normal'))} — Bus {row.get('bus', row.get('Bus', 'N/A'))}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Fault Prob.", fmt(row.get("fault_prob", 0)))
            c2.metric("Congestion Prob.", fmt(row.get("congestion_prob", 0)))
            c3.metric("Voltage Prob.", fmt(row.get("voltage_prob", 0)))
            st.caption(f"Recommended action: `{row.get('recommended_action', 'monitor')}`")
            if "explanation" in row:
                with st.expander("Explanation"):
                    st.write(row.get("explanation", "No explanation available."))
            st.divider()

        with st.expander("Full recommendations table"):
            safe_df(display, height=360)
    else:
        missing("Copilot recommendations missing. Run: python scripts/run_phase6_copilot.py")


# ============================================================
# TAB 6 — RL MITIGATION
# ============================================================
with tab6:
    st.subheader("🛡️ RL Mitigation Recommendation Policy")

    section_card("<b>Framing:</b> The DQN is evaluated as a sequential recommendation policy, not closed-loop physical grid control.")

    if table4 is not None and len(table4) > 0:
        safe_df(table4)

        policy_col = find_col(table4, ["Policy", "policy"])
        reward_col = find_col(table4, ["Mean Reward", "mean_reward", "reward"])

        if policy_col and reward_col:
            plot = table4.copy()
            plot[reward_col] = to_num(plot[reward_col])
            fig = px.bar(plot, x=policy_col, y=reward_col, text=reward_col, title="RL Policy Reward Comparison")
            fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            safe_plotly(fig)
    else:
        missing("RL performance missing. Run: python scripts/run_phase8_final_results.py")

    if table6 is not None and len(table6) > 0:
        action_col = find_col(table6, ["Action", "action"])
        count_col = find_col(table6, ["Count", "count"])
        if action_col and count_col:
            fig = px.bar(table6, x=action_col, y=count_col, color=action_col, text=count_col, title="Converged DQN Action Distribution")
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            safe_plotly(fig)


# ============================================================
# TAB 7 — CASCADING RISK
# ============================================================
with tab7:
    st.subheader("🌊 Cascading Risk Simulation")

    section_card("Simulation uses illustrative propagation parameters. Quantitative RL mitigation is reported separately in the RL Mitigation tab.")

    if cas_summary is not None:
        safe_df(cas_summary)
        try:
            metric_col = "Metric"
            value_col = "Value"
            if metric_col in cas_summary.columns and value_col in cas_summary.columns:
                plot = cas_summary.copy()
                plot[value_col] = to_num(plot[value_col])
                fig = px.bar(plot, x=metric_col, y=value_col, text=value_col, title="Cascading Risk Summary")
                fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                safe_plotly(fig)
        except Exception as e:
            st.warning(f"Could not plot cascading summary: {e}")
    else:
        missing("Cascading summary missing. Run: python scripts/run_phase10_cascading_simulation.py")

    if cas_series is not None:
        time_col = find_col(cas_series, ["timestep", "time", "step"])
        risk_col = find_col(cas_series, ["cascading_risk", "risk", "Risk"])
        if time_col and risk_col:
            fig = px.line(cas_series, x=time_col, y=risk_col, title="Cascading Risk Over Time")
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            safe_plotly(fig)
    else:
        missing("Cascading time series missing. Run: python scripts/run_phase10_cascading_simulation.py")


# ============================================================
# TAB 8 — STRESS BENCHMARK
# ============================================================
with tab8:
    st.subheader("📈 Stress-Regime Performance Benchmark")

    section_card(
        "<b>Benchmark framing:</b> This is stress-regime performance, not out-of-distribution robustness. "
        "Temporal models remain ahead of static baselines across renewable stress regimes, but no single temporal architecture dominates all stress levels."
    )

    if stress_summary is not None and len(stress_summary) > 0:
        s = stress_summary.copy()
        if "model" in s.columns:
            s["Model Clean"] = s["model"].apply(clean_model_name)
        elif "Model" in s.columns:
            s["Model Clean"] = s["Model"].apply(clean_model_name)

        stress_col = find_col(s, ["stress_scale", "Stress Scale", "alpha", "α"])
        auprc_s = find_col(s, ["macro_auprc_mean", "Macro AUPRC", "macro_auprc"])
        recall_s = find_col(s, ["macro_recall_mean", "Macro Recall", "macro_recall"])

        if stress_col and auprc_s:
            fig = px.line(s, x=stress_col, y=auprc_s, color="Model Clean", markers=True, title="Macro AUPRC Across Renewable Stress Regimes", color_discrete_map=MODEL_COLORS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            safe_plotly(fig)

        if stress_col and recall_s:
            fig = px.line(s, x=stress_col, y=recall_s, color="Model Clean", markers=True, title="Macro Recall Across Renewable Stress Regimes", color_discrete_map=MODEL_COLORS)
            fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            safe_plotly(fig)

        with st.expander("Stress benchmark summary table"):
            safe_df(s)
    else:
        missing("Phase 11 stress benchmark missing. Run: python scripts/run_phase11_stress_benchmark.py")

    if stress_best is not None:
        st.markdown("#### Best Model by Stress Level")
        safe_df(stress_best)

    if stress_meta is not None:
        st.markdown("#### Dataset Metadata")
        safe_df(stress_meta)


# ============================================================
# TAB 9 — LINKEDIN SNAPSHOT
# ============================================================
with tab9:
    st.subheader("AI Grid Resilience Copilot")

    left, right = st.columns([1.05, 1])

    with left:
        st.markdown(
            "**Predictive fault, congestion, and voltage-instability precursor intelligence "
            "for renewable-rich distribution feeders.**"
        )

        k1, k2 = st.columns(2)
        k3, k4 = st.columns(2)

        k1.metric("Macro F1", "92%+")
        k2.metric("Macro AUPRC", "96%+")
        k3.metric("CPU Inference", "31.3 ms")
        k4.metric("RL Reward Gain", "+7.8%")

        st.markdown(
            "**From reactive grid monitoring to predictive, explainable, "
            "AI-assisted resilience intelligence.**"
        )

        st.caption(
            "Temporal graph learning • Renewable-aware resilience • Explainable AI for smart distribution grids"
        )

    with right:
        fig_path = PHASE4 / "attention_feeder.png"

        if fig_path.exists() and Image is not None:
            try:
                st.image(Image.open(fig_path), width="stretch")
            except TypeError:
                st.image(Image.open(fig_path), use_container_width=True)
        else:
            st.warning(
                "Attention feeder image missing. Run: python scripts/run_phase4_explainability.py"
            )

    st.markdown("### Suggested LinkedIn Caption")

    st.code(
        "From reactive grid monitoring to predictive, explainable, AI-assisted resilience intelligence.\n\n"
        "Built an AI Grid Resilience Copilot using temporal graph neural networks, renewable-aware stress modeling, "
        "attention-based explainability, and RL-assisted operator recommendations for smart distribution grids.\n\n"
        "Validated through multi-seed benchmarking, renewable ablation, explainability analysis, stress-regime benchmarking, "
        "and cascading-risk simulations.",
        language="text",
    )

    st.info("Use Windows Snipping Tool to capture the top compact panel for LinkedIn.")