"""
Home.py
=======
Streamlit Early Warning Dashboard for Cyber Attack Prediction & Risk Analytics.

Provides real-time visualization of network flow risks, threat levels,
explainable AI top indicators, and batch CSV flow analysis.
"""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.auth import require_auth
from app.db import list_users, get_recent_alerts, log_alert, log_prediction

# ---------------------------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cyber Threat Early Warning System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enforce authentication on entry; stops page execution if not logged in
current_user = require_auth()

# ---------------------------------------------------------------------------
# Custom CSS for Modern Dark Glassmorphic Cybersecurity Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Dark Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
    }

    /* Threat Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.18);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.25);
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.18);
        color: #f97316;
        border: 1px solid rgba(249, 115, 22, 0.5);
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.25);
    }
    .badge-medium {
        background: rgba(234, 179, 8, 0.18);
        color: #eab308;
        border: 1px solid rgba(234, 179, 8, 0.5);
    }
    .badge-low {
        background: rgba(16, 185, 129, 0.18);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }

    /* Alert Banners */
    .alert-banner-active {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(185, 28, 28, 0.12) 100%);
        border: 1px solid rgba(239, 68, 68, 0.6);
        border-left: 6px solid #ef4444;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(239, 68, 68, 0.15);
    }
    .alert-banner-nominal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.08) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        border-left: 6px solid #10b981;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
    }

    /* Indicator Cards */
    .indicator-card {
        background: rgba(30, 41, 59, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .indicator-card:hover {
        border-color: rgba(56, 189, 248, 0.5);
        transform: translateX(4px);
    }

    /* Metric Containers */
    .metric-box {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f8fafc;
        margin-top: 4px;
    }
    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Temporary Simulation Function (Fake Predictions)
# Later this function will be replaced by `src.predict.predict(df)`
# ---------------------------------------------------------------------------

def get_fake_prediction(flow_data: Optional[Dict[str, Any]] = None, scenario: str = "Random") -> Dict[str, Any]:
    """Temporary function returning fake prediction telemetry in the required schema.

    Schema:
        - risk_score: float (0.0 to 100.0)
        - threat_level: str ("Low", "Medium", "High", "Critical")
        - attack_probability: float (0.0 to 1.0)
        - anomaly_score: float (0.0 to 1.0)
        - top_indicators: list of str
        - alert: bool
        - priority: str ("Low", "Medium", "High", "Critical")

    *Note*: This will be seamlessly replaced by `src.predict.predict()` in future iterations.
    """
    if scenario == "Critical DDoS Flood":
        risk = 88.5
        threat = "Critical"
        prob = 0.94
        anom = 0.82
        alert = True
        prio = "Critical"
        indicators = [
            "Severe packet transmission rate (72,400 pkts/s), signaling an active volumetric flood.",
            "Elevated byte throughput (12.4 MB/s), indicating severe network bandwidth saturation.",
            "Asymmetric forward-to-backward packet ratio (14.2), characteristic of non-responsive flood targets.",
        ]
    elif scenario == "Stealth Port Scan":
        risk = 68.2
        threat = "High"
        prob = 0.76
        anom = 0.58
        alert = True
        prio = "High"
        indicators = [
            "Spike in TCP Reset (RST) flag density (31.4%), caused by closed-port sweeps.",
            "Extremely low flow duration (12 ms) consistent with automated port scanner probes.",
            "Minimal packet payload size (44.0 bytes), typical of SYN reconnaissance packets.",
        ]
    elif scenario == "Medium Anomaly":
        risk = 42.0
        threat = "Medium"
        prob = 0.38
        anom = 0.65
        alert = False
        prio = "Medium"
        indicators = [
            "Unusual flow longevity tier (tier 3), deviating from typical baseline interactive sessions.",
            "Moderate divergence in SYN/ACK flag frequencies (ratio 2.1).",
            "Slight elevation in forward packet share (68.4%).",
        ]
    elif scenario == "Benign Traffic":
        risk = 14.3
        threat = "Low"
        prob = 0.05
        anom = 0.12
        alert = False
        prio = "Low"
        indicators = [
            "Balanced bidirectional packet flow ratio (1.05), typical of standard HTTP/HTTPS browsing.",
            "Normal packet payload size variance (mean 412 bytes) matching verified application traffic.",
            "Clean TCP handshake teardown with zero abnormal RST or URG control flags.",
        ]
    else:
        # Dynamic randomized or row-driven generation
        if flow_data and "pkt_rate" in flow_data:
            # Synthetic score heuristic based on input
            pkt = float(flow_data.get("pkt_rate", 500))
            prob = min(0.99, max(0.05, pkt / 20000.0))
            anom = random.uniform(0.1, 0.8)
        else:
            prob = random.choice([0.12, 0.45, 0.78, 0.92])
            anom = random.uniform(0.15, 0.75)

        raw_risk = round(100.0 * (0.60 * prob + 0.25 * anom + 0.15 * random.uniform(0.1, 0.6)), 1)
        risk = min(100.0, max(0.0, raw_risk))

        if risk <= 25.0:
            threat, prio, alert = "Low", "Low", False
        elif risk <= 50.0:
            threat, prio, alert = "Medium", "Medium", False
        elif risk <= 75.0:
            threat, prio, alert = "High", "High", True
        else:
            threat, prio, alert = "Critical", "Critical", True

        indicators = [
            f"Volumetric telemetry indicates anomaly index score of {prob * 100:.1f}%.",
            f"Unsupervised Isolation Forest path divergence measured at {anom:.2f}.",
            "Flag ratio telemetry deviates from normal operational baseline distributions.",
        ]

    return {
        "risk_score": float(risk),
        "threat_level": threat,
        "attack_probability": round(prob, 4),
        "anomaly_score": round(anom, 4),
        "top_indicators": indicators,
        "alert": alert,
        "priority": prio,
    }


# ---------------------------------------------------------------------------
# UI Helper: Threat Level Badge HTML
# ---------------------------------------------------------------------------

def render_threat_badge(threat_level: str) -> str:
    """Return styled HTML badge for a threat level."""
    cls_map = {
        "Critical": "badge badge-critical",
        "High": "badge badge-high",
        "Medium": "badge badge-medium",
        "Low": "badge badge-low",
    }
    badge_cls = cls_map.get(threat_level, "badge badge-low")
    return f'<span class="{badge_cls}">● {threat_level} Threat</span>'


# ---------------------------------------------------------------------------
# UI Helper: Plotly Risk Gauge
# ---------------------------------------------------------------------------

def create_risk_gauge(risk_score: float) -> go.Figure:
    """Create an interactive Plotly gauge chart for the risk score."""
    # Color based on threat zone
    if risk_score <= 25:
        bar_color = "#10b981"  # Emerald Green
    elif risk_score <= 50:
        bar_color = "#eab308"  # Amber Yellow
    elif risk_score <= 75:
        bar_color = "#f97316"  # Orange
    else:
        bar_color = "#ef4444"  # Crimson Red

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "/100", "font": {"size": 36, "color": "#f8fafc", "family": "Inter"}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#64748b",
                    "tickfont": {"size": 11, "color": "#94a3b8"},
                },
                "bar": {"color": bar_color, "thickness": 0.28},
                "bgcolor": "rgba(30, 41, 59, 0.4)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": "rgba(16, 185, 129, 0.12)"},
                    {"range": [25, 50], "color": "rgba(234, 179, 8, 0.12)"},
                    {"range": [50, 75], "color": "rgba(249, 115, 22, 0.14)"},
                    {"range": [75, 100], "color": "rgba(239, 68, 68, 0.18)"},
                ],
                "threshold": {
                    "line": {"color": "#ef4444", "width": 3},
                    "thickness": 0.85,
                    "value": 60.0,  # Alert threshold marker
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin={"l": 25, "r": 25, "t": 20, "b": 15},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"},
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar: Controls & Live Simulation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")
    scenario_choice = st.selectbox(
        "Demonstration Threat Scenario",
        [
            "Critical DDoS Flood",
            "Stealth Port Scan",
            "Medium Anomaly",
            "Benign Traffic",
            "Random Telemetry",
        ],
        index=0,
    )

    alert_thresh_slider = st.slider(
        "Alert Trigger Threshold",
        min_value=30,
        max_value=90,
        value=60,
        step=5,
        help="Risk score threshold (default 60) that automatically raises early warning alerts.",
    )

    st.markdown("---")
    st.markdown("### ℹ️ Engine Information")
    st.caption(
        "**Multi-Model Fusion Formula**:\n"
        "$$\\text{Risk} = 100 \\times (0.60 P_{\\text{atk}} + 0.25 S_{\\text{anom}} + 0.15 C_{\\text{ctx}})$$"
    )
    st.caption("Powered by XGBoost + Isolation Forest + SHAP Explainability.")


# ---------------------------------------------------------------------------
# Header Section
# ---------------------------------------------------------------------------

col_title, col_status = st.columns([3, 1])

with col_title:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; color: #f8fafc;">
                🛡️ Cyber Attack Prediction & Early Warning System
            </h1>
        </div>
        <p style="color: #94a3b8; font-size: 1.05rem; margin-top: 0; margin-bottom: 1.5rem;">
            Real-time behavioral intrusion detection, anomaly scoring, and explainable threat triage.
        </p>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 1: CSV Upload Widget
# ---------------------------------------------------------------------------

st.markdown("### 📂 Ingest Network Traffic Telemetry")

uploaded_file = st.file_uploader(
    "Upload Network Flow CSV (CIC-IDS2017 schema or processed features)",
    type=["csv"],
    help="Upload network flow telemetry records to evaluate risk, anomaly scores, and trigger early warnings.",
)

active_flow_data: Optional[Dict[str, Any]] = None
uploaded_df: Optional[pd.DataFrame] = None

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded `{uploaded_file.name}` with {len(uploaded_df):,} flow records.")
        with st.expander("🔍 Preview Ingested Network Flows", expanded=False):
            st.dataframe(uploaded_df.head(10), use_container_width=True)

        selected_row_idx = st.slider(
            "Select Flow Record Index for Live Threat Inspection",
            min_value=0,
            max_value=len(uploaded_df) - 1,
            value=0,
            step=1,
        )
        active_flow_data = uploaded_df.iloc[selected_row_idx].to_dict()
    except Exception as exc:
        st.error(f"Error parsing CSV file: {exc}")

# Generate prediction (calls fake prediction function, which later connects to src/predict.predict)
pred = get_fake_prediction(flow_data=active_flow_data, scenario=scenario_choice)

# Dynamic override if alert threshold slider adjusted
if pred["risk_score"] >= alert_thresh_slider:
    pred["alert"] = True

# Automatically persist every prediction and alert to the SQLite database
try:
    log_prediction(pred, username=current_user.get("username", "analyst"))
    if pred["alert"]:
        log_alert(
            risk_score=pred["risk_score"],
            threat_level=pred["threat_level"],
            attack_probability=pred["attack_probability"],
            source=f"Live Sensor ({current_user.get('username', 'analyst')})",
            status="Open",
            notes=" | ".join(pred.get("top_indicators", [])),
        )
except Exception as log_err:
    pass

# ---------------------------------------------------------------------------
# Section 2: Early Warning Status Banner
# ---------------------------------------------------------------------------

if pred["alert"]:
    st.markdown(
        f"""
        <div class="alert-banner-active">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.8rem;">🚨</span>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #fca5a5; letter-spacing: 0.02em;">
                            EARLY WARNING ALERT ACTIVE: Potential Cyber Attack Detected
                        </div>
                        <div style="font-size: 0.9rem; color: #fecaca; margin-top: 2px;">
                            Risk Score ({pred['risk_score']:.1f}/100) exceeded threshold ({alert_thresh_slider}). Immediate security triage recommended.
                        </div>
                    </div>
                </div>
                <div>
                    <span class="badge badge-critical">PRIORITY: {pred['priority']}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="alert-banner-nominal">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 1.6rem;">🛡️</span>
                    <div>
                        <div style="font-size: 1.05rem; font-weight: 700; color: #6ee7b7;">
                            SYSTEM NOMINAL: No Threat Alerts Triggered
                        </div>
                        <div style="font-size: 0.88rem; color: #a7f3d0; margin-top: 2px;">
                            Current network telemetry is operating within verified baseline anomaly tolerances.
                        </div>
                    </div>
                </div>
                <div>
                    <span class="badge badge-low">STATUS: HEALTHY</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 3: Live Threat Telemetry & Risk Metrics
# ---------------------------------------------------------------------------

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("#### 🎯 Composite Risk Gauge")
    
    # Header with Threat Badge
    col_badge, col_prio = st.columns([1, 1])
    with col_badge:
        st.markdown(render_threat_badge(pred["threat_level"]), unsafe_allow_html=True)
    with col_prio:
        st.markdown(
            f"<div style='text-align: right; color: #94a3b8; font-size: 0.9rem;'>Investigation Priority: <strong style='color: #f8fafc;'>{pred['priority']}</strong></div>",
            unsafe_allow_html=True,
        )

    # Plotly Gauge Chart
    fig_gauge = create_risk_gauge(pred["risk_score"])
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Underlying Telemetry Metric Cards
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">Attack Probability</div>
                <div class="metric-val" style="color: #f59e0b;">{pred['attack_probability'] * 100:.1f}%</div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">Supervised XGBoost Weight (60%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-lbl">Anomaly Score</div>
                <div class="metric-val" style="color: #06b6d4;">{pred['anomaly_score'] * 100:.1f}%</div>
                <div style="color: #94a3b8; font-size: 0.75rem; margin-top: 4px;">Unsupervised Baseline Weight (25%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown("#### 🧠 Top Risk Indicators (Explainable AI)")
    st.caption("SHAP feature attribution explaining the primary drivers behind this assessment:")

    for idx, indicator in enumerate(pred["top_indicators"], 1):
        st.markdown(
            f"""
            <div class="indicator-card">
                <div style="display: flex; align-items: flex-start; gap: 10px;">
                    <span style="color: #38bdf8; font-weight: 800; font-size: 1.05rem;">#{idx}</span>
                    <span style="color: #e2e8f0; font-size: 0.92rem; line-height: 1.45;">{indicator}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚡ Risk Engine Weights Distribution")

    # Mini breakdown progress bars
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.4); border-radius: 8px; padding: 14px; border: 1px solid rgba(148, 163, 184, 0.1);">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 4px;">
                <span>Attack Probability Contribution</span>
                <strong>{0.60 * pred['attack_probability'] * 100:.1f} pts</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e1; margin-bottom: 4px;">
                <span>Anomaly Divergence Contribution</span>
                <strong>{0.25 * pred['anomaly_score'] * 100:.1f} pts</strong>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; color: #cbd5e1;">
                <span>Behavioral Context Telemetry</span>
                <strong>{max(0.0, pred['risk_score'] - (0.60 * pred['attack_probability'] * 100 + 0.25 * pred['anomaly_score'] * 100)):.1f} pts</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Section 4: Batch Simulation Table (When CSV Uploaded)
# ---------------------------------------------------------------------------

if uploaded_df is not None:
    st.markdown("---")
    st.markdown("### 📊 Batch Telemetry Overview")
    
    # Process up to first 25 rows for demonstration
    batch_rows = []
    sample_slice = uploaded_df.head(25)
    for i in range(len(sample_slice)):
        r = sample_slice.iloc[i].to_dict()
        res = get_fake_prediction(flow_data=r, scenario="Random")
        batch_rows.append({
            "Flow Index": i,
            "Risk Score": res["risk_score"],
            "Threat Level": res["threat_level"],
            "Attack Prob": f"{res['attack_probability']*100:.1f}%",
            "Anomaly Score": f"{res['anomaly_score']*100:.1f}%",
            "Alert Triggered": "🚨 YES" if res["alert"] else "✅ No",
            "Priority": res["priority"],
        })

    batch_df = pd.DataFrame(batch_rows)
    st.dataframe(batch_df, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 5: Administrator Control Panel (Admin Role Only)
# ---------------------------------------------------------------------------
if current_user.get("role") == "admin":
    st.markdown("---")
    st.markdown("### 👑 SOC Administrator Control & Audit Panel")
    admin_tab1, admin_tab2 = st.tabs(["👥 User Registry", "📋 Audit & Alert Logs"])

    with admin_tab1:
        st.caption("Active operators and their assigned access roles in the SQLite security database:")
        users_list = list_users()
        if users_list:
            st.dataframe(pd.DataFrame(users_list), use_container_width=True)

    with admin_tab2:
        st.caption("Recent system security alerts recorded in the database:")
        alerts_list = get_recent_alerts(limit=25)
        if alerts_list:
            st.dataframe(pd.DataFrame(alerts_list), use_container_width=True)
        else:
            st.info("No recorded alerts yet in the database.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Cyber Attack Prediction & Early Warning System | Built with Streamlit, Plotly, SQLAlchemy & bcrypt."
)
