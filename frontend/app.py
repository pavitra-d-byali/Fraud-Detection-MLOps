"""
Fraud Detection — Streamlit Frontend
Calls the FastAPI backend at BACKEND_URL (default: http://localhost:8000)
"""

import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Fraud Detection | MLOps",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — dark accent strip + minor polish
# ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* --- global font & background --- */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* --- top colour bar --- */
        header[data-testid="stHeader"] {
            background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
            height: 4px;
        }

        /* --- metric cards --- */
        [data-testid="stMetric"] {
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 12px;
            padding: 16px 20px;
        }
        [data-testid="stMetricLabel"] > div { font-size: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase; opacity: 0.7; }

        /* --- sidebar --- */
        [data-testid="stSidebar"] { background: #0f1117; }

        /* --- predict button --- */
        div[data-testid="stForm"] button[kind="primaryFormSubmit"],
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #6366f1, #06b6d4);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        div[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover { opacity: 0.88; }

        /* --- feature expander --- */
        details summary { font-weight: 600; }

        /* hide default hamburger menu */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []  # list of prediction dicts

# ──────────────────────────────────────────────────────────────
# Helper: call backend
# ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def fetch_health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=4)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def call_predict(payload: dict) -> dict:
    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


# ──────────────────────────────────────────────────────────────
# Sidebar — model health & metadata
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ FraudShield MLOps")
    st.markdown("---")

    health = fetch_health()

    if "error" in health:
        st.error(f"**Backend offline**\n\n`{health['error']}`")
        threshold = None
        cost_fp = cost_fn = None
    else:
        st.success("**Backend — Online ✓**")
        threshold = health.get("threshold")
        cost_fp = health.get("cost_policy", {}).get("cost_fp")
        cost_fn = health.get("cost_policy", {}).get("cost_fn")

        st.markdown("#### Model Metadata")
        st.markdown(f"- **Model:** XGBoost Classifier")
        st.markdown(f"- **Threshold:** `{threshold:.4f}`")
        st.markdown(f"- **Cost FP / FN:** `{cost_fp}` / `{cost_fn}`")
        st.markdown(f"- **Decision logic:** `prob ≥ threshold → FRAUD`")

    st.markdown("---")
    st.markdown("#### ℹ️ Dataset Info")
    st.caption(
        "V1–V28 are PCA-transformed features from raw card transactions. "
        "Values are standardised (μ=0, σ≈1). Time is seconds since first transaction."
    )

    st.markdown("---")
    if st.button("🔄 Refresh Health Check"):
        fetch_health.clear()
        st.rerun()

    if st.session_state.history:
        st.markdown("---")
        st.markdown(f"#### 📊 Session Stats")
        total = len(st.session_state.history)
        frauds = sum(1 for h in st.session_state.history if h["decision"] == "FRAUD")
        st.metric("Total Predictions", total)
        st.metric("Fraud Detected", frauds, delta=f"{frauds/total*100:.0f}%")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

# ──────────────────────────────────────────────────────────────
# Main page header
# ──────────────────────────────────────────────────────────────
st.markdown("# 🔍 Transaction Risk Analyser")
st.caption(
    "Cost-sensitive fraud detection · XGBoost + FastAPI · "
    f"Threshold optimised to minimise `{cost_fp}·FP + {cost_fn}·FN`"
    if cost_fp is not None
    else "Cost-sensitive fraud detection · XGBoost + FastAPI"
)

tab_predict, tab_history, tab_about = st.tabs(["🎯 Predict", "📜 History", "📖 About"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab_predict:

    # Quick-fill buttons
    col_l, col_f, col_r, col_rnd, _ = st.columns([1, 1.2, 0.8, 0.8, 3])
    load_legit = col_l.button("✅ Legit Sample")
    load_fraud = col_f.button("🚨 Suspicious Sample")
    do_clear   = col_r.button("🧹 Clear")
    do_random  = col_rnd.button("🎲 Random")

    st.markdown("---")

    # Default values
    def default_vals(mode="zero"):
        vals = {"Time": 0.0, "Amount": 0.0}
        for i in range(1, 29):
            vals[f"V{i}"] = 0.0

        if mode == "legit":
            vals.update({"Time": 10000.0, "Amount": 25.50,
                         "V3": 0.12, "V7": -0.08, "V14": 0.05})
        elif mode == "fraud":
            vals.update({"Time": 50000.0, "Amount": 2500.0,
                         "V1": -3.5, "V2": 2.8, "V3": -4.2,
                         "V4": 3.1, "V10": -2.4, "V12": -1.7,
                         "V14": -3.8, "V17": -2.1})
        elif mode == "random":
            import random, math
            vals["Time"]   = round(random.uniform(0, 172792), 2)
            vals["Amount"] = round(abs(random.gauss(50, 80)), 2)
            for i in range(1, 29):
                vals[f"V{i}"] = round(random.gauss(0, 1), 4)
        return vals

    # Determine which mode to populate
    if "form_vals" not in st.session_state:
        st.session_state.form_vals = default_vals("zero")

    if load_legit:
        st.session_state.form_vals = default_vals("legit")
    elif load_fraud:
        st.session_state.form_vals = default_vals("fraud")
    elif do_clear:
        st.session_state.form_vals = default_vals("zero")
    elif do_random:
        st.session_state.form_vals = default_vals("random")

    fv = st.session_state.form_vals

    # ── Form ─────────────────────────────────────────────────
    with st.form("predict_form"):

        # Transaction metadata
        st.markdown("##### Transaction Metadata")
        c1, c2 = st.columns(2)
        time_val   = c1.number_input("Time (seconds since first tx)", value=float(fv["Time"]),   step=1.0, format="%.2f", key="inp_time")
        amount_val = c2.number_input("Amount (transaction value)",    value=float(fv["Amount"]), step=0.01, format="%.2f", key="inp_amount")

        # V features — grouped in an expander to reduce visual clutter
        st.markdown("##### Encoded Risk Features (V1 – V28)")
        st.caption("PCA-anonymised components. All values standardised.")

        v_vals = {}
        with st.expander("🔽 Show / Edit V-Features", expanded=True):
            cols_per_row = 7
            v_keys = [f"V{i}" for i in range(1, 29)]
            rows = [v_keys[i : i + cols_per_row] for i in range(0, 28, cols_per_row)]
            for row in rows:
                cols = st.columns(len(row))
                for col, name in zip(cols, row):
                    v_vals[name] = col.number_input(
                        name,
                        value=float(fv[name]),
                        step=0.01,
                        format="%.4f",
                        key=f"inp_{name}",
                    )

        submitted = st.form_submit_button("⚡ Analyse Transaction", use_container_width=True, type="primary")

    # ── Prediction ───────────────────────────────────────────
    if submitted:
        payload = {"Time": time_val, "Amount": amount_val, **v_vals}

        with st.spinner("Running inference…"):
            try:
                result = call_predict(payload)

                prob      = result["fraud_probability"]
                decision  = result["decision"]
                thr       = result["threshold"]
                cp        = result.get("cost_policy", {})

                # Store in history
                st.session_state.history.append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "decision":  decision,
                    "probability": prob,
                    "threshold": thr,
                    "amount": amount_val,
                })

                # ── Verdict banner ───────────────────────────
                is_fraud = decision == "FRAUD"
                if is_fraud:
                    st.error(f"## 🚨 FRAUD DETECTED  —  {prob*100:.2f}% probability")
                else:
                    st.success(f"## ✅ LEGITIMATE  —  {prob*100:.2f}% probability")

                # ── Metrics row ──────────────────────────────
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Decision",          decision)
                m2.metric("Fraud Probability", f"{prob:.4f}")
                m3.metric("Threshold",         f"{thr:.4f}")
                m4.metric("Risk Level",
                           "HIGH 🔴" if prob >= 0.75 else
                           "MEDIUM 🟡" if prob >= 0.25 else
                           "LOW 🟢")

                # ── Plotly gauge ─────────────────────────────
                gauge_color = "#ef4444" if prob >= thr else "#22c55e"

                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=round(prob * 100, 2),
                    number={"suffix": "%", "font": {"size": 42}},
                    delta={"reference": thr * 100, "suffix": "% vs threshold",
                           "increasing": {"color": "#ef4444"},
                           "decreasing": {"color": "#22c55e"}},
                    title={"text": "Fraud Probability", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1,
                                 "tickcolor": "rgba(255,255,255,0.3)"},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "rgba(255,255,255,0.04)",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0,   25],  "color": "rgba(34,197,94,0.15)"},
                            {"range": [25,  75],  "color": "rgba(234,179,8,0.12)"},
                            {"range": [75,  100], "color": "rgba(239,68,68,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "#a78bfa", "width": 3},
                            "thickness": 0.82,
                            "value": thr * 100,
                        },
                    },
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="white",
                    height=300,
                    margin=dict(t=40, b=10, l=40, r=40),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Cost policy context ───────────────────────
                if cp:
                    st.info(
                        f"**Cost Policy** · False Positive cost = **{cp.get('cost_fp')}** | "
                        f"False Negative cost = **{cp.get('cost_fn')}**\n\n"
                        f"The decision threshold `{thr:.4f}` was selected to minimise expected "
                        f"loss = `{cp.get('cost_fp')}·FP + {cp.get('cost_fn')}·FN` on the hold-out set."
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Cannot reach backend. Make sure FastAPI is running:\n\n"
                    "```bash\nuvicorn src.api.app:app --reload\n```"
                )
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")


# ══════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════
with tab_history:
    if not st.session_state.history:
        st.info("No predictions yet. Run a prediction in the **Predict** tab.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        df_hist.index = range(1, len(df_hist) + 1)
        df_hist.index.name = "#"

        # Summary metrics
        total = len(df_hist)
        n_fraud = (df_hist["decision"] == "FRAUD").sum()
        avg_prob = df_hist["probability"].mean()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", total)
        c2.metric("Fraud Flagged",     n_fraud,
                  delta=f"{n_fraud/total*100:.0f}% of total",
                  delta_color="inverse")
        c3.metric("Avg Fraud Prob",    f"{avg_prob:.4f}")

        st.markdown("---")

        # Colour-coded dataframe
        def highlight_decision(row):
            colour = "background-color:#3f1414; color:#f87171;" if row["decision"] == "FRAUD" \
                else "background-color:#0f2d1a; color:#86efac;"
            return [colour] * len(row)

        st.dataframe(
            df_hist.style.apply(highlight_decision, axis=1),
            use_container_width=True,
        )

        # Probability trend chart
        st.markdown("#### Fraud Probability — Prediction Trend")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=list(range(1, total + 1)),
            y=df_hist["probability"].tolist(),
            mode="lines+markers",
            name="Fraud Prob",
            line=dict(color="#6366f1", width=2),
            marker=dict(
                color=["#ef4444" if d == "FRAUD" else "#22c55e"
                       for d in df_hist["decision"]],
                size=10, line=dict(color="white", width=1),
            ),
        ))
        if threshold:
            fig2.add_hline(
                y=threshold, line_dash="dot",
                line_color="#a78bfa", line_width=2,
                annotation_text=f"Threshold {threshold:.4f}",
                annotation_font_color="#a78bfa",
            )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font_color="white",
            xaxis_title="Prediction #",
            yaxis_title="Fraud Probability",
            yaxis=dict(range=[0, 1]),
            height=320,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        """
        ## 📖 Project Overview

        ### Architecture
        | Layer | Technology |
        |---|---|
        | ML Model | XGBoost (binary:logistic) |
        | Backend API | FastAPI + Uvicorn |
        | Frontend | Streamlit |
        | Experiment Tracking | MLflow |
        | Containerisation | Docker |

        ### Training Pipeline
        1. **Data** — [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
           284 807 transactions · 492 frauds (0.17% class imbalance)
        2. **Preprocessing** — PCA anonymisation (V1–V28) + Amount/Time scaling  
        3. **Cross-validation** — 5-fold Stratified KFold  
        4. **Model** — XGBoost with `scale_pos_weight` to handle imbalance  
        5. **Threshold tuning** — Grid search over `[0.01, 0.5]` to minimise  
           `cost_fp · |FP| + cost_fn · |FN|` on hold-out set

        ### Cost Policy
        | Label | Cost | Rationale |
        |---|---|---|
        | False Positive (blocking legit tx) | 1 | Customer friction |
        | False Negative (missing fraud) | 50 | Financial loss |

        ### MLOps Features
        - **MLflow** — metric + parameter logging, model registry  
        - **GitHub Actions CI** — installs dependencies + smoke test on every push  
        - **Docker** — fully containerised backend  
        - **Absolute path safety** — all scripts work regardless of working directory

        ### Running Locally
        ```bash
        # 1. Train model (requires creditcard.csv in data/)
        python src/model/train.py

        # 2. Start backend API
        uvicorn src.api.app:app --reload

        # 3. Start Streamlit frontend (separate terminal)
        streamlit run frontend/app.py
        ```
        """
    )
