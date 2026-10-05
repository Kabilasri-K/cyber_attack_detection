"""
1_Alerts.py
===========
Security Alerts Management & Operational Incident Triage.

Features:
- Filterable security alerts table by Threat Level and Status.
- Real-time Priority Incident Queue sorted by (risk_score * attack_probability).
- Interactive alert status transitions (Open -> Investigating -> Resolved).
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
from Backend.db import get_alerts, update_alert_status, init_db

# ---------------------------------------------------------------------------
# Page Setup & Authentication
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Active Alerts | SOC Incident Triage",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

current_user = require_auth()
init_db()

# ---------------------------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .queue-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .queue-card-critical {
        border-left: 5px solid #ef4444;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.12);
    }
    .queue-card-high {
        border-left: 5px solid #f97316;
        box-shadow: 0 4px 15px rgba(249, 115, 22, 0.12);
    }
    .queue-card-medium {
        border-left: 5px solid #eab308;
    }
    .queue-card-low {
        border-left: 5px solid #10b981;
    }
    .badge {
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        display: inline-block;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.5); }
    .badge-high { background: rgba(249, 115, 22, 0.2); color: #f97316; border: 1px solid rgba(249, 115, 22, 0.5); }
    .badge-medium { background: rgba(234, 179, 8, 0.2); color: #eab308; border: 1px solid rgba(234, 179, 8, 0.5); }
    .badge-low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.5); }
    .badge-status-open { background: rgba(239, 68, 68, 0.15); color: #fca5a5; }
    .badge-status-inv { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; }
    .badge-status-res { background: rgba(16, 185, 129, 0.15); color: #6ee7b7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header & Key Summary Metrics
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 6px;">
        <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
            🚨 Security Alerts & Incident Triage
        </h1>
    </div>
    <p style="color: #94a3b8; font-size: 1.0rem; margin-top: 0; margin-bottom: 1.5rem;">
        Prioritized threat queue sorted by expected impact (Risk × Probability) and operational response console.
    </p>
    """,
    unsafe_allow_html=True,
)

# Fetch all alerts from database
raw_alerts = get_alerts(limit=500)

if not raw_alerts:
    st.info("No security alerts currently recorded in the database. Head over to the Home dashboard to generate telemetry.")
    st.stop()

df_alerts = pd.DataFrame(raw_alerts)

# Ensure numeric types
df_alerts["risk_score"] = df_alerts["risk_score"].astype(float)
df_alerts["attack_probability"] = df_alerts["attack_probability"].fillna(0.5).astype(float)

# Compute priority score: risk_score * attack_probability
df_alerts["priority_score"] = (df_alerts["risk_score"] * df_alerts["attack_probability"]).round(2)

# Metric Summary Cards
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Total Recorded Alerts", len(df_alerts))
with col_m2:
    open_count = len(df_alerts[df_alerts["status"] == "Open"])
    st.metric("Unresolved Open Alerts", open_count, delta=f"{open_count} active", delta_color="inverse")
with col_m3:
    crit_count = len(df_alerts[df_alerts["threat_level"].isin(["Critical", "High"])])
    st.metric("Critical / High Severity", crit_count)
with col_m4:
    max_prio = df_alerts["priority_score"].max() if not df_alerts.empty else 0.0
    st.metric("Peak Priority Score", f"{max_prio:.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 1: Priority Queue Sorted by (Risk * Probability)
# ---------------------------------------------------------------------------
st.markdown("### ⚡ Priority Incident Queue (Sorted by Risk × Probability)")
st.caption(
    "Incidents ranked in descending order of expected attack severity: "
    "$$\\text{Priority Score} = \\text{Risk Score} \\times P(\\text{Attack})$$"
)

# Filter to active/open/investigating incidents for the priority queue
active_alerts = df_alerts[df_alerts["status"].isin(["Open", "Investigating"])].copy()
if active_alerts.empty:
    active_alerts = df_alerts.copy()

queue_sorted = active_alerts.sort_values(by="priority_score", ascending=False).head(5)

for rank, (_, row) in enumerate(queue_sorted.iterrows(), 1):
    level = str(row["threat_level"])
    card_cls = f"queue-card queue-card-{level.lower()}"
    badge_cls = f"badge badge-{level.lower()}"

    status_str = str(row["status"])
    st_cls = "badge-status-open" if status_str == "Open" else "badge-status-inv"

    st.markdown(
        f"""
        <div class="{card_cls}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 8px;">
                <div>
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
                        <strong style="color: #38bdf8; font-size: 1.1rem;">#{rank} Priority</strong>
                        <span class="{badge_cls}">● {level}</span>
                        <span class="badge {st_cls}">Status: {status_str}</span>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Alert ID #{row['id']}</span>
                    </div>
                    <div style="color: #f8fafc; font-size: 0.95rem; font-weight: 600; margin-top: 4px;">
                        Source Sensor: <span style="font-family: monospace; color: #a5f3fc;">{row['source']}</span>
                    </div>
                    <div style="color: #cbd5e1; font-size: 0.88rem; margin-top: 4px;">
                        {row.get('notes') or 'Behavioral anomaly detected exceeding alert trigger threshold.'}
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Priority Score</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #f87171;">{row['priority_score']:.1f}</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">
                        Risk {row['risk_score']:.1f} &bull; Prob {row['attack_probability']*100:.0f}%
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Section 2: Filterable Alerts Table
# ---------------------------------------------------------------------------
st.markdown("### 📋 Filterable Alert Inventory")

# Filter Controls
col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    filter_threat = st.selectbox(
        "Filter by Threat Level",
        ["All", "Critical", "High", "Medium", "Low"],
        index=0,
    )

with col_f2:
    filter_status = st.selectbox(
        "Filter by Resolution Status",
        ["All", "Open", "Investigating", "Resolved"],
        index=0,
    )

with col_f3:
    sort_option = st.selectbox(
        "Sort By",
        ["Priority Score (Risk × Prob)", "Newest Timestamp", "Risk Score (High to Low)"],
        index=0,
    )

# Apply filters
filtered_df = df_alerts.copy()
if filter_threat != "All":
    filtered_df = filtered_df[filtered_df["threat_level"] == filter_threat]
if filter_status != "All":
    filtered_df = filtered_df[filtered_df["status"] == filter_status]

# Apply sorting
if sort_option == "Priority Score (Risk × Prob)":
    filtered_df = filtered_df.sort_values(by="priority_score", ascending=False)
elif sort_option == "Newest Timestamp":
    filtered_df = filtered_df.sort_values(by="timestamp", ascending=False)
else:
    filtered_df = filtered_df.sort_values(by="risk_score", ascending=False)

# Display Table
display_cols = [
    "id",
    "timestamp",
    "threat_level",
    "priority_score",
    "risk_score",
    "attack_probability",
    "source",
    "status",
    "notes",
]

st.dataframe(
    filtered_df[display_cols].rename(
        columns={
            "id": "Alert ID",
            "timestamp": "Timestamp",
            "threat_level": "Threat Level",
            "priority_score": "Priority Score",
            "risk_score": "Risk",
            "attack_probability": "Attack Prob",
            "source": "Source",
            "status": "Status",
            "notes": "Investigation Notes",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------------------------
# Section 3: Interactive Incident Action & Status Update
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🛠️ Incident Triage Action Console")

with st.expander("Update Alert Status & Resolution Notes", expanded=False):
    with st.form("alert_status_form"):
        col_act1, col_act2, col_act3 = st.columns(3)
        with col_act1:
            target_id = st.selectbox(
                "Target Alert ID",
                options=df_alerts["id"].tolist(),
                help="Select the alert ID to triage.",
            )
        with col_act2:
            new_status = st.selectbox(
                "Set New Status",
                ["Open", "Investigating", "Resolved"],
            )
        with col_act3:
            operator_name = st.text_input(
                "Assigned Analyst",
                value=current_user.get("username", "analyst"),
            )

        resolution_notes = st.text_area(
            "Triage & Mitigation Notes",
            placeholder="Document threat mitigation, IP firewall block, or triage steps...",
        )

        submit_action = st.form_submit_button("Update Incident Status", type="primary")

        if submit_action:
            success = update_alert_status(
                alert_id=int(target_id),
                new_status=new_status,
                resolved_by=operator_name,
                notes=resolution_notes if resolution_notes.strip() else None,
            )
            if success:
                st.success(f"Alert #{target_id} updated to '{new_status}' by {operator_name}!")
                st.rerun()
            else:
                st.error(f"Failed to update Alert #{target_id}.")
