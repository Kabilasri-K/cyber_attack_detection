"""
risk_engine.py
==============
Dynamic Cybersecurity Risk Scoring and Alerting Engine.

Calculates composite risk scores, maps to operational threat levels,
and triggers real-time early warning alerts based on probabilistic,
anomaly, and behavioral network telemetry.

Core Formula:
-------------
risk = 100 * (0.6 * attack_probability + 0.25 * normalized_anomaly_score + 0.15 * context_factor)

Threat Level Mapping:
---------------------
-  0 to 25  : Low
- 26 to 50  : Medium
- 51 to 75  : High
- 76 to 100 : Critical

Alert Triggering:
-----------------
Alert fires (True) if:
  1. Current risk score exceeds configurable threshold (default: >= 60.0), OR
  2. Risk score is rising sharply compared to the previous observation window
     (default delta: >= 20.0).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default Constants
# ---------------------------------------------------------------------------

DEFAULT_ALERT_THRESHOLD: float = 60.0
DEFAULT_SHARP_RISE_THRESHOLD: float = 20.0

WEIGHT_ATTACK_PROB: float = 0.60
WEIGHT_ANOMALY_SCORE: float = 0.25
WEIGHT_CONTEXT_FACTOR: float = 0.15


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def normalize_anomaly_score(raw_anomaly: float | np.ndarray) -> float | np.ndarray:
    """Normalize raw Isolation Forest anomaly score into the [0.0, 1.0] interval.

    Isolation Forest score_samples returns values typically in [-0.8, -0.3].
    Negative score_samples (-score) centers around [0.35, 0.70].
    We apply a sigmoid-like or min-max calibration to map higher anomaly into [0, 1].
    """
    arr = np.asarray(raw_anomaly, dtype=float)
    # If the score is already in [0, 1], simply clip
    if np.all((arr >= 0.0) & (arr <= 1.0)):
        norm = np.clip(arr, 0.0, 1.0)
    else:
        # Logistic calibration centered around 0.50
        norm = 1.0 / (1.0 + np.exp(-10.0 * (arr - 0.50)))
        norm = np.clip(norm, 0.0, 1.0)

    return float(norm) if np.ndim(raw_anomaly) == 0 else norm


def compute_context_factor(features: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> float | np.ndarray:
    """Compute behavioral context factor in [0.0, 1.0] from network telemetry.

    Cybersecurity context:
    Evaluates indicators of active attack behaviors:
    1. Unusually high packet rate (DoS / flood indicator).
    2. Elevated reset flag ratio (RST) from port scan attempts against closed ports.
    3. Severe SYN-to-ACK asymmetry or excessive SYN flag density (SYN flood probe).
    4. Elevated byte transmission rate (data exfiltration or bulk flood).
    """
    if isinstance(features, pd.DataFrame):
        return np.array([compute_context_factor(row) for _, row in features.iterrows()])

    # Extract relevant indicators with safe fallbacks
    pkt_rate = float(features.get("pkt_rate", 0.0))
    rst_ratio = float(features.get("rst_flag_ratio", 0.0))
    syn_ratio = float(features.get("syn_flag_ratio", 0.0))
    syn_ack_ratio = float(features.get("syn_to_ack_ratio", 0.0))
    byte_rate = float(features.get("byte_rate", 0.0))

    scores: List[float] = []

    # 1. Packet rate stress (normal interactive < 1,000 pkts/s; floods > 10,000 pkts/s)
    if pkt_rate > 0:
        # Scale logarithmically: 100 pkts/s -> ~0.1, 10,000 pkts/s -> ~0.8, 50,000+ -> 1.0
        pkt_score = np.clip((np.log10(max(1.0, pkt_rate)) - 2.0) / 3.0, 0.0, 1.0)
        scores.append(float(pkt_score))

    # 2. RST flag anomalies (port scanning trigger)
    if rst_ratio > 0.02:
        rst_score = np.clip(rst_ratio / 0.20, 0.0, 1.0)
        scores.append(float(rst_score))

    # 3. SYN / ACK imbalance (SYN flood trigger)
    if syn_ratio > 0.10 or syn_ack_ratio > 2.0:
        syn_score = np.clip(max(syn_ratio / 0.50, (syn_ack_ratio - 1.0) / 5.0), 0.0, 1.0)
        scores.append(float(syn_score))

    # 4. Byte rate volumetric surge (> 5 MB/s)
    if byte_rate > 100_000:
        byte_score = np.clip((np.log10(max(1.0, byte_rate)) - 4.0) / 3.0, 0.0, 1.0)
        scores.append(float(byte_score))

    if not scores:
        return 0.0

    # Composite context: blend max severity with mean behavioral stress
    composite = 0.60 * max(scores) + 0.40 * (sum(scores) / len(scores))
    return float(np.clip(composite, 0.0, 1.0))


def map_threat_level(risk_score: float) -> str:
    """Map a numerical risk score (0-100) to operational threat levels.

    Scale:
    -  0 to 25  : Low
    - 26 to 50  : Medium
    - 51 to 75  : High
    - 76 to 100 : Critical
    """
    score = float(np.clip(risk_score, 0.0, 100.0))
    if score <= 25.0:
        return "Low"
    elif score <= 50.0:
        return "Medium"
    elif score <= 75.0:
        return "High"
    else:
        return "Critical"


def map_priority(risk_score: float, is_alert: bool) -> str:
    """Map risk score and alert state to operational incident triage priority."""
    score = float(risk_score)
    if score >= 76.0:
        return "Critical"
    elif score >= 51.0 or is_alert:
        return "High"
    elif score >= 26.0:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Core Risk Scoring Engine
# ---------------------------------------------------------------------------

def calculate_risk(
    attack_probability: float,
    anomaly_score: float,
    context_factor: Optional[float] = None,
    features: Optional[Dict[str, Any]] = None,
    prev_risk: Optional[float] = None,
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
    sharp_rise_threshold: float = DEFAULT_SHARP_RISE_THRESHOLD,
) -> Dict[str, Any]:
    """Calculate composite risk, threat level, and alert status for a network event.

    Formula:
        risk = 100 * (0.60 * attack_prob + 0.25 * norm_anomaly + 0.15 * context_factor)

    Parameters
    ----------
    attack_probability : float
        Supervised classifier probability of attack [0.0, 1.0].
    anomaly_score : float
        Unsupervised anomaly detection score.
    context_factor : float, optional
        Pre-computed context factor [0.0, 1.0]. If None, calculated from `features`.
    features : dict, optional
        Feature telemetry dictionary used to compute context_factor if not provided.
    prev_risk : float, optional
        Previous window risk score to detect sharp rises.
    alert_threshold : float, default=60.0
        Threshold above which an alert is automatically fired.
    sharp_rise_threshold : float, default=20.0
        Surge delta above previous window that triggers an early warning alert.

    Returns
    -------
    dict
        {
            "risk_score": float (0-100),
            "threat_level": str ("Low"|"Medium"|"High"|"Critical"),
            "alert": bool,
            "priority": str,
            "attack_probability": float,
            "anomaly_score": float,
            "context_factor": float
        }
    """
    # 1. Normalize attack probability
    p_attack = float(np.clip(attack_probability, 0.0, 1.0))

    # 2. Normalize anomaly score
    norm_anomaly = float(normalize_anomaly_score(anomaly_score))

    # 3. Determine context factor
    if context_factor is not None:
        ctx = float(np.clip(context_factor, 0.0, 1.0))
    elif features is not None:
        ctx = float(compute_context_factor(features))
    else:
        ctx = 0.0

    # 4. Compute composite risk formula
    raw_risk = 100.0 * (
        WEIGHT_ATTACK_PROB * p_attack
        + WEIGHT_ANOMALY_SCORE * norm_anomaly
        + WEIGHT_CONTEXT_FACTOR * ctx
    )
    risk_score = round(float(np.clip(raw_risk, 0.0, 100.0)), 2)

    # 5. Map to Threat Level
    threat_level = map_threat_level(risk_score)

    # 6. Evaluate Alert Conditions
    sharp_rise = False
    if prev_risk is not None:
        delta = risk_score - float(prev_risk)
        if delta >= sharp_rise_threshold:
            sharp_rise = True

    is_alert = bool((risk_score >= alert_threshold) or sharp_rise)

    # 7. Priority
    priority = map_priority(risk_score, is_alert)

    return {
        "risk_score": risk_score,
        "threat_level": threat_level,
        "alert": is_alert,
        "sharp_rise": sharp_rise,
        "priority": priority,
        "attack_probability": round(p_attack, 4),
        "anomaly_score": round(norm_anomaly, 4),
        "context_factor": round(ctx, 4),
    }


class RiskEngine:
    """Stateful Risk Engine supporting temporal window tracking for trend analysis."""

    def __init__(
        self,
        alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
        sharp_rise_threshold: float = DEFAULT_SHARP_RISE_THRESHOLD,
    ):
        self.alert_threshold = alert_threshold
        self.sharp_rise_threshold = sharp_rise_threshold
        self.history: List[float] = []

    def evaluate(
        self,
        attack_probability: float,
        anomaly_score: float,
        features: Optional[Dict[str, Any]] = None,
        context_factor: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate current event against historical window."""
        prev_risk = self.history[-1] if self.history else None

        result = calculate_risk(
            attack_probability=attack_probability,
            anomaly_score=anomaly_score,
            context_factor=context_factor,
            features=features,
            prev_risk=prev_risk,
            alert_threshold=self.alert_threshold,
            sharp_rise_threshold=self.sharp_rise_threshold,
        )

        self.history.append(result["risk_score"])
        # Maintain rolling window of last 100 events
        if len(self.history) > 100:
            self.history.pop(0)

        return result
