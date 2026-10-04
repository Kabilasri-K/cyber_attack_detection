"""
2_Trends.py
===========
Threat Trends & Historical Telemetry Analytics.

Features:
- Plotly interactive line chart: Risk score trajectory over time with threshold lines and rolling averages.
- Plotly bar & donut charts: Attack type distribution and classification counts.
- Threat level severity breakdown.
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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.auth import require_auth
from app.db import get_predictions, init_db

# ---------------------------------------------------------------------------
# Page Setup & Authentication
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Threat Trends | Analytics",
    page_icon="📈",
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
            📈 Threat Trends & Historical Telemetry Analytics
        </h1>
    </div>
    <p style="color: #94a3b8; font-size: 1.0rem; margin-top: 0; margin-bottom: 1.5rem;">
        Temporal risk trajectories, attack category frequencies, and behavioral pattern distributions.
    </p>
    """,
    unsafe_allow_html=True,
)

# Fetch historical prediction logs from database
predictions = get_predictions(limit=1000)

if not predictions:
    st.info("No prediction telemetry logs found in the database. Run predictions on the Home page first.")
    st.stop()

df = pd.DataFrame(predictions)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df["risk_score"] = df["risk_score"].astype(float)
df["attack_probability"] = df["attack_probability"].astype(float)

# Summary Key Performance Indicators
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Historical Records", f"{len(df):,}")
with c2:
    st.metric("Average Risk Score", f"{df['risk_score'].mean():.1f} / 100")
with c3:
    st.metric("Peak Recorded Risk", f"{df['risk_score'].max():.1f}")
with c4:
    total_attacks = len(df[df["threat_level"].isin(["High", "Critical"])])
    st.metric("High/Critical Incidents", total_attacks)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Chart 1: Risk Score Over Time (Interactive Plotly Timeline)
# ---------------------------------------------------------------------------
st.markdown("### ⏱️ Risk Score Trajectory Over Time")
st.caption("Temporal fluctuation of network flow risk with alert threshold and 5-point moving average trendline.")

# Compute moving average
df["moving_avg"] = df["risk_score"].rolling(window=5, min_periods=1).mean()

fig_timeline = go.Figure()

# Threat level color mapping
color_map = {
    "Critical": "#ef4444",
    "High": "#f97316",
    "Medium": "#eab308",
    "Low": "#10b981",
}

# 1. Individual event markers
for level, group in df.groupby("threat_level"):
    fig_timeline.add_trace(
        go.Scatter(
            x=group["timestamp"],
            y=group["risk_score"],
            mode="markers",
            name=f"{level} Risk Event",
            marker={
                "size": 9,
                "color": color_map.get(level, "#94a3b8"),
                "opacity": 0.85,
                "line": {"width": 1, "color": "#ffffff"},
            },
            text=group.apply(
                lambda r: f"<b>{r['predicted_attack_type']}</b><br>Risk: {r['risk_score']:.1f}<br>Prob: {r['attack_probability']*100:.1f}%",
                axis=1,
            ),
            hoverinfo="text+x",
        )
    )

# 2. Moving average trendline
fig_timeline.add_trace(
    go.Scatter(
        x=df["timestamp"],
        y=df["moving_avg"],
        mode="lines",
        name="Rolling Trend (MA-5)",
        line={"color": "#38bdf8", "width": 2.5, "shape": "spline"},
    )
)

# 3. Alert threshold boundary line (at risk = 60)
fig_timeline.add_hline(
    y=60.0,
    line_dash="dash",
    line_color="#ef4444",
    annotation_text="Early Warning Threshold (60.0)",
    annotation_position="bottom right",
    annotation_font_color="#fca5a5",
)

fig_timeline.update_layout(
    height=400,
    margin={"l": 40, "r": 25, "t": 25, "b": 40},
    paper_bgcolor="rgba(15, 23, 42, 0.4)",
    plot_bgcolor="rgba(15, 23, 42, 0.6)",
    font={"color": "#f8fafc", "family": "Inter"},
    xaxis={
        "gridcolor": "rgba(148, 163, 184, 0.1)",
        "title": "Observation Timestamp",
    },
    yaxis={
        "gridcolor": "rgba(148, 163, 184, 0.1)",
        "title": "Composite Risk Score (0 - 100)",
        "range": [0, 105],
    },
    legend={
        "orientation": "h",
        "yanchor": "bottom",
        "y": 1.02,
        "xanchor": "right",
        "x": 1,
    },
)

st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Attack Type Counts & Severity Breakdown
# ---------------------------------------------------------------------------
col_bar, col_pie = st.columns([1.2, 1], gap="large")

with col_bar:
    st.markdown("### 📊 Attack Type Classification Counts")
    st.caption("Volume of network flows categorized by classifier attack signature:")

    attack_counts = (
        df["predicted_attack_type"]
        .value_counts()
        .reset_index()
    )
    attack_counts.columns = ["Attack Type", "Flow Count"]

    fig_bar = px.bar(
        attack_counts,
        x="Attack Type",
        y="Flow Count",
        color="Attack Type",
        color_discrete_sequence=["#38bdf8", "#ef4444", "#f97316", "#a855f7", "#10b981"],
        text="Flow Count",
    )

    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        height=350,
        margin={"l": 30, "r": 20, "t": 20, "b": 40},
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        font={"color": "#f8fafc"},
        xaxis={"gridcolor": "rgba(148, 163, 184, 0.1)"},
        yaxis={"gridcolor": "rgba(148, 163, 184, 0.1)"},
        showlegend=False,
    )

    st.plotly_chart(fig_bar, use_container_width=True)

with col_pie:
    st.markdown("### 🍩 Threat Level Distribution")
    st.caption("Proportion of flows categorized across severity bands:")

    threat_counts = df["threat_level"].value_counts().reset_index()
    threat_counts.columns = ["Threat Level", "Count"]

    fig_donut = px.pie(
        threat_counts,
        names="Threat Level",
        values="Count",
        hole=0.55,
        color="Threat Level",
        color_discrete_map=color_map,
    )

    fig_donut.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="rgba(15, 23, 42, 0.4)",
        font={"color": "#f8fafc"},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.1},
    )

    st.plotly_chart(fig_donut, use_container_width=True)
