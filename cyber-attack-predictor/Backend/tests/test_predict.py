"""
test_predict.py
===============
Unit and integration tests for the prediction pipeline (src/predict.py),
risk engine (src/risk_engine.py), and SHAP explanations (src/explain.py).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backend.src.predict import predict
from Backend.src.risk_engine import calculate_risk, map_threat_level, map_priority, compute_context_factor
from Backend.src.explain import explain_instance


@pytest.fixture
def sample_feature_data() -> pd.DataFrame:
    """Load sample feature records from features.parquet if available, else synthetic dataframe."""
    features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    if features_path.exists():
        df = pd.read_parquet(features_path)
        return df.head(5).copy()

    # Synthetic fallback fixture
    data = {
        "pkt_rate": [150.0, 55000.0, 12000.0],
        "byte_rate": [45000.0, 8000000.0, 1500000.0],
        "fwd_pkt_rate": [100.0, 50000.0, 10000.0],
        "bwd_pkt_rate": [50.0, 5000.0, 2000.0],
        "fwd_bwd_pkt_ratio": [2.0, 10.0, 5.0],
        "fwd_pkt_ratio": [0.66, 0.91, 0.83],
        "bwd_pkt_ratio": [0.34, 0.09, 0.17],
        "syn_flag_ratio": [0.01, 0.45, 0.05],
        "ack_flag_ratio": [0.80, 0.02, 0.10],
        "rst_flag_ratio": [0.0, 0.01, 0.35],
        "fin_flag_ratio": [0.05, 0.0, 0.0],
        "psh_flag_ratio": [0.10, 0.30, 0.05],
        "urg_flag_ratio": [0.0, 0.0, 0.0],
        "syn_to_ack_ratio": [0.01, 22.5, 0.5],
        "avg_pkt_size": [300.0, 1450.0, 125.0],
        "avg_fwd_pkt_size": [450.0, 1450.0, 150.0],
        "avg_bwd_pkt_size": [150.0, 0.0, 100.0],
        "flow_duration_bucket": [2, 0, 1],
        "log_flow_duration": [14.2, 8.5, 11.1],
    }
    return pd.DataFrame(data)


def test_predict_returns_required_keys(sample_feature_data):
    """Verify that predict(df) returns a list of dictionaries with all required fields."""
    predictions = predict(sample_feature_data)

    assert isinstance(predictions, list)
    assert len(predictions) == len(sample_feature_data)

    required_keys = {
        "risk_score",
        "threat_level",
        "attack_probability",
        "anomaly_score",
        "top_indicators",
        "alert",
        "priority",
    }

    for record in predictions:
        assert isinstance(record, dict)
        assert required_keys.issubset(record.keys()), f"Missing keys in {record.keys()}"


def test_predict_value_ranges_and_types(sample_feature_data):
    """Verify that output metrics obey their valid mathematical bounds and types."""
    predictions = predict(sample_feature_data)

    valid_threat_levels = {"Low", "Medium", "High", "Critical"}
    valid_priorities = {"Low", "Medium", "High", "Critical"}

    for rec in predictions:
        # Risk Score: 0 to 100
        assert 0.0 <= rec["risk_score"] <= 100.0, f"Invalid risk score: {rec['risk_score']}"

        # Threat Level
        assert rec["threat_level"] in valid_threat_levels

        # Probabilities and Anomaly Scores: 0.0 to 1.0
        assert 0.0 <= rec["attack_probability"] <= 1.0
        assert 0.0 <= rec["anomaly_score"] <= 1.0

        # Top Indicators: List of strings
        assert isinstance(rec["top_indicators"], list)
        for ind in rec["top_indicators"]:
            assert isinstance(ind, str)
            assert len(ind) > 10

        # Alert: boolean
        assert isinstance(rec["alert"], bool)

        # Priority
        assert rec["priority"] in valid_priorities


def test_risk_formula_calculation():
    """Verify exact formula calculation: risk = 100 * (0.6*prob + 0.25*anom + 0.15*context)."""
    p_attack = 0.80
    norm_anom = 0.60
    ctx = 0.50

    # Expected: 100 * (0.60*0.80 + 0.25*0.60 + 0.15*0.50)
    # 100 * (0.48 + 0.15 + 0.075) = 100 * 0.705 = 70.5
    expected_risk = 70.50

    result = calculate_risk(
        attack_probability=p_attack,
        anomaly_score=norm_anom,
        context_factor=ctx,
        alert_threshold=60.0,
    )

    assert result["risk_score"] == pytest.approx(expected_risk, abs=0.1)
    assert result["threat_level"] == "High"
    assert result["alert"] is True  # 70.5 >= 60.0


def test_threat_level_boundaries():
    """Verify edge boundaries for threat level categorization."""
    assert map_threat_level(0.0) == "Low"
    assert map_threat_level(25.0) == "Low"
    assert map_threat_level(25.1) == "Medium"
    assert map_threat_level(50.0) == "Medium"
    assert map_threat_level(50.1) == "High"
    assert map_threat_level(75.0) == "High"
    assert map_threat_level(75.1) == "Critical"
    assert map_threat_level(100.0) == "Critical"


def test_sharp_rise_alert_triggering():
    """Verify that a sharp surge in risk compared to previous window triggers an alert."""
    # Low absolute risk (e.g. 40), but jumped by 30 points from previous window (10 -> 40)
    result = calculate_risk(
        attack_probability=0.35,
        anomaly_score=0.40,
        context_factor=0.20,
        prev_risk=10.0,
        alert_threshold=60.0,
        sharp_rise_threshold=20.0,
    )

    assert result["sharp_rise"] is True
    assert result["alert"] is True  # Alert fired due to sharp surge even though risk < 60


def test_predict_single_dict_input(sample_feature_data):
    """Verify that predict() accepts a single dictionary input."""
    single_record = sample_feature_data.iloc[0].to_dict()
    preds = predict(single_record)

    assert isinstance(preds, list)
    assert len(preds) == 1
    assert "risk_score" in preds[0]
    assert "threat_level" in preds[0]


def test_predict_empty_input():
    """Verify that passing an empty DataFrame returns an empty list."""
    empty_df = pd.DataFrame()
    preds = predict(empty_df)
    assert preds == []


def test_explain_instance_top_3(sample_feature_data):
    """Verify that explain_instance produces top 3 plain-English sentences."""
    row = sample_feature_data.iloc[0].to_dict()
    explanations = explain_instance(row, top_k=3)

    assert isinstance(explanations, list)
    assert len(explanations) == 3
    for sentence in explanations:
        assert isinstance(sentence, str)
        assert len(sentence.split()) >= 4  # A real sentence with multiple words
