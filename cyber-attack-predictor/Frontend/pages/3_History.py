"""
3_History.py
============
Prediction Audit Log & Historical Telemetry Registry.

Features:
- Reads historical prediction records directly from SQLite database.
- Multi-dimensional filtering by Threat Level, Attack Type, and Alert Trigger.
- Full SHAP explanation drilldown inspector.
- Export historical logs to CSV for security compliance and audit trails.
- Protected by require_auth().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from Frontend.auth import require_auth
from Backend.db import get_predictions, init_db

# ---------------------------------------------------------------------------
# Page Setup & Authentication
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Prediction History | Audit Log",
    page_icon="📜",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_user = require_auth()
init_db()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
            📜 Prediction Audit Log & Telemetry History
        </h1>
    </div>
    <p style="color: #94a3b8; font-size: 1.0rem; margin-top: 0; margin-bottom: 1.5rem;">
        Persistent database registry of all evaluated network flows, risk assessments, and analyst sessions.
    </p>
    """,
    unsafe_allow_html=True,
)

# Fetch historical predictions from SQLite
records = get_predictions(limit=1000)

if not records:
    st.info("No prediction records found in the database. Run predictions on the Home page first.")
    st.stop()

df = pd.DataFrame(records)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ---------------------------------------------------------------------------
# Filter Controls
# ---------------------------------------------------------------------------
st.markdown("### 🔍 Query & Filter Telemetry Records")

col_flt1, col_flt2, col_flt3, col_flt4 = st.columns(4)

with col_flt1:
    threat_opts = ["All"] + sorted(df["threat_level"].unique().tolist())
    sel_threat = st.selectbox("Threat Level", threat_opts, index=0)

with col_flt2:
    attack_opts = ["All"] + sorted(df["predicted_attack_type"].dropna().unique().tolist())
    sel_attack = st.selectbox("Predicted Attack Type", attack_opts, index=0)

with col_flt3:
    sel_alert = st.selectbox("Alert State", ["All", "Alerts Only (Triggered)", "Nominal (No Alert)"], index=0)

with col_flt4:
    search_query = st.text_input("Search Indicators / Operator", placeholder="e.g. DoS, flood, admin...")

# Apply Filters
filtered = df.copy()

if sel_threat != "All":
    filtered = filtered[filtered["threat_level"] == sel_threat]

if sel_attack != "All":
    filtered = filtered[filtered["predicted_attack_type"] == sel_attack]

if sel_alert == "Alerts Only (Triggered)":
    filtered = filtered[filtered["alert"] == True]
elif sel_alert == "Nominal (No Alert)":
    filtered = filtered[filtered["alert"] == False]

if search_query.strip():
    q = search_query.strip().lower()
    mask = (
        filtered["top_indicators"].fillna("").str.lower().str.contains(q)
        | filtered["analyzed_by"].fillna("").str.lower().str.contains(q)
        | filtered["predicted_attack_type"].fillna("").str.lower().str.contains(q)
    )
    filtered = filtered[mask]

# ---------------------------------------------------------------------------
# Summary Header & Export Action
# ---------------------------------------------------------------------------
col_cnt, col_exp = st.columns([3, 1])

with col_cnt:
    st.markdown(f"Displaying **{len(filtered):,}** of **{len(df):,}** total historical records.")

with col_exp:
    csv_data = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered History (CSV)",
        data=csv_data,
        file_name="prediction_history_audit.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Data Table
# ---------------------------------------------------------------------------
display_df = filtered.copy()
display_df["attack_probability"] = (display_df["attack_probability"] * 100).round(1).astype(str) + "%"
display_df["anomaly_score"] = (display_df["anomaly_score"] * 100).round(1).astype(str) + "%"
display_df["alert"] = display_df["alert"].apply(lambda a: "🚨 YES" if a else "✅ No")

table_cols = [
    "id",
    "timestamp",
    "threat_level",
    "risk_score",
    "attack_probability",
    "anomaly_score",
    "predicted_attack_type",
    "alert",
    "priority",
    "analyzed_by",
]

st.dataframe(
    display_df[table_cols].rename(
        columns={
            "id": "Record ID",
            "timestamp": "Timestamp",
            "threat_level": "Threat Level",
            "risk_score": "Risk (/100)",
            "attack_probability": "Attack Prob",
            "anomaly_score": "Anomaly Score",
            "predicted_attack_type": "Class Signature",
            "alert": "Alert",
            "priority": "Priority",
            "analyzed_by": "Operator",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Drilldown Inspector: Full SHAP Attribution Details
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🔬 Flow Record Deep-Dive Inspector")

with st.expander("Inspect Full Explainability & Indicator Details for a Record", expanded=False):
    sel_id = st.selectbox(
        "Select Record ID to Inspect",
        options=filtered["id"].tolist() if not filtered.empty else [0],
    )

    if sel_id and not filtered.empty:
        rec = filtered[filtered["id"] == sel_id].iloc[0]

        d_col1, d_col2 = st.columns([1, 1.5], gap="large")
        with d_col1:
            st.markdown(f"#### Summary for Record #{rec['id']}")
            st.markdown(f"- **Timestamp**: `{rec['timestamp']}`")
            st.markdown(f"- **Evaluated By**: `{rec.get('analyzed_by') or 'Automated Sensor'}`")
            st.markdown(f"- **Predicted Signature**: **{rec['predicted_attack_type']}**")
            st.markdown(f"- **Threat Level**: `{rec['threat_level']}`")
            st.markdown(f"- **Risk Score**: `{rec['risk_score']:.2f}` / 100")
            st.markdown(f"- **Incident Priority**: `{rec['priority']}`")

        with d_col2:
            st.markdown("#### Explainable AI Indicators (Top Drivers)")
            indicators_raw = rec.get("top_indicators")
            if indicators_raw:
                lines = [line.strip() for line in str(indicators_raw).split("\n") if line.strip()]
                for idx, line in enumerate(lines, 1):
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 6px; padding: 10px 14px; margin-bottom: 8px;">
                            <strong style="color: #38bdf8;">#{idx}</strong> <span style="color: #f1f5f9;">{line}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No specific text indicators logged for this entry.")
