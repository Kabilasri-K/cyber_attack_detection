"""
predict.py
==========
Inference pipeline for the Cyber Attack Prediction & Early Warning System.

Exposes a unified predict(df) interface that coordinates:
1. Supervised attack classification (XGBoost)
2. Unsupervised zero-day anomaly detection (Isolation Forest)
3. Dynamic risk scoring and early warning alerts (risk_engine)
4. Model explainability via natural language feature attributions (explain)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.explain import explain_prediction
from src.risk_engine import (
    DEFAULT_ALERT_THRESHOLD,
    DEFAULT_SHARP_RISE_THRESHOLD,
    calculate_risk,
    compute_context_factor,
    normalize_anomaly_score,
)

# ---------------------------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
XGB_PATH = MODELS_DIR / "xgboost_attack_classifier.joblib"
ISO_PATH = MODELS_DIR / "isolation_forest_detector.joblib"
META_PATH = MODELS_DIR / "model_metadata.joblib"


# ---------------------------------------------------------------------------
# Model Loader Cache
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Lazy-loaded model cache to avoid repeated disk I/O across predictions."""

    _xgb_model = None
    _iso_model = None
    _metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def load(cls) -> None:
        """Load serialised models and metadata if not already cached."""
        if cls._xgb_model is None:
            if not XGB_PATH.exists():
                raise FileNotFoundError(
                    f"XGBoost model missing at {XGB_PATH}. Run 'python src/train.py' first."
                )
            cls._xgb_model = joblib.load(XGB_PATH)

        if cls._iso_model is None:
            if not ISO_PATH.exists():
                raise FileNotFoundError(
                    f"Isolation Forest model missing at {ISO_PATH}. Run 'python src/train.py' first."
                )
            cls._iso_model = joblib.load(ISO_PATH)

        if cls._metadata is None:
            if META_PATH.exists():
                cls._metadata = joblib.load(META_PATH)
            else:
                cls._metadata = {}

    @classmethod
    def get_xgb(cls):
        cls.load()
        return cls._xgb_model

    @classmethod
    def get_iso(cls):
        cls.load()
        return cls._iso_model

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        cls.load()
        return cls._metadata or {}


# ---------------------------------------------------------------------------
# Core Prediction Interface
# ---------------------------------------------------------------------------

def predict(
    df: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]],
    alert_threshold: float = DEFAULT_ALERT_THRESHOLD,
    prev_risks: Optional[List[float]] = None,
) -> List[Dict[str, Any]]:
    """Generate risk assessment and threat predictions for network telemetry records.

    Parameters
    ----------
    df : pd.DataFrame, dict, or list of dicts
        Input network flow telemetry records containing engineered features.
    alert_threshold : float, default=60.0
        Risk threshold above which an alert is triggered.
    prev_risks : list of float, optional
        Previous window risk scores to detect sudden threat surges.

    Returns
    -------
    list of dict
        A list of prediction dictionaries, each containing:
        - risk_score: float (0.0 to 100.0)
        - threat_level: str ("Low", "Medium", "High", "Critical")
        - attack_probability: float (0.0 to 1.0)
        - anomaly_score: float (0.0 to 1.0)
        - top_indicators: list of str (plain-English sentences driving prediction)
        - alert: bool (True if risk exceeds threshold or rises sharply)
        - priority: str ("Low", "Medium", "High", "Critical")
        - predicted_attack_type: str (e.g. "BENIGN", "DoS Hulk", "PortScan")
    """
    # 1. Format input into a pandas DataFrame
    if isinstance(df, dict):
        input_df = pd.DataFrame([df])
    elif isinstance(df, list):
        input_df = pd.DataFrame(df)
    elif isinstance(df, pd.DataFrame):
        input_df = df.copy()
    else:
        raise TypeError(f"Expected DataFrame, dict, or list of dicts, got {type(df)}")

    if input_df.empty:
        return []

    # 2. Retrieve cached models and metadata
    xgb_model = ModelRegistry.get_xgb()
    iso_model = ModelRegistry.get_iso()
    metadata = ModelRegistry.get_metadata()

    expected_features: List[str] = metadata.get("feature_names", [])
    label_mapping: Dict[int, str] = metadata.get(
        "label_mapping", {0: "BENIGN", 1: "DoS Hulk", 2: "PortScan"}
    )

    # 3. Align DataFrame columns with expected feature matrix
    if expected_features:
        for feat in expected_features:
            if feat not in input_df.columns:
                input_df[feat] = 0.0
        X = input_df[expected_features].copy()
    else:
        X = input_df.copy()

    # Fill any potential remaining NaN / Inf values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 4. Supervised Prediction (XGBoost)
    xgb_proba = xgb_model.predict_proba(X)
    xgb_preds = xgb_model.predict(X)

    # Determine attack probability:
    # Class 0 is assumed BENIGN; attack probability is sum of all non-benign classes
    if xgb_proba.shape[1] > 1:
        benign_prob = xgb_proba[:, 0]
        attack_probs = np.clip(1.0 - benign_prob, 0.0, 1.0)
    else:
        attack_probs = xgb_proba[:, 0]

    # 5. Unsupervised Anomaly Detection (Isolation Forest)
    # raw score: lower is more anomalous; negate to make higher more anomalous
    raw_anomaly_scores = -iso_model.score_samples(X)
    normalized_anomalies = normalize_anomaly_score(raw_anomaly_scores)

    # 6. Behavioral Context Factor & Explainability
    context_factors = compute_context_factor(X)
    top_indicators_batch = explain_prediction(X, top_k=3)

    # 7. Synthesize Composite Risk and Assemble Output
    results: List[Dict[str, Any]] = []

    for i in range(len(input_df)):
        p_attack = float(attack_probs[i])
        norm_anom = float(normalized_anomalies[i])
        ctx_val = float(context_factors[i])
        pred_label = label_mapping.get(int(xgb_preds[i]), f"Class_{xgb_preds[i]}")

        prev_risk = None
        if prev_risks is not None and i < len(prev_risks):
            prev_risk = prev_risks[i]

        risk_info = calculate_risk(
            attack_probability=p_attack,
            anomaly_score=norm_anom,
            context_factor=ctx_val,
            prev_risk=prev_risk,
            alert_threshold=alert_threshold,
        )

        indicators = top_indicators_batch[i] if i < len(top_indicators_batch) else []

        record = {
            "risk_score": risk_info["risk_score"],
            "threat_level": risk_info["threat_level"],
            "attack_probability": risk_info["attack_probability"],
            "anomaly_score": risk_info["anomaly_score"],
            "top_indicators": indicators,
            "alert": risk_info["alert"],
            "priority": risk_info["priority"],
            "predicted_attack_type": pred_label,
        }
        results.append(record)

    return results


# ---------------------------------------------------------------------------
# CLI / Quick Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.features import INPUT_PATH

    print("=" * 60)
    print("Testing predict() interface on processed feature samples...")
    print("=" * 60)

    features_path = PROJECT_ROOT / "data" / "processed" / "features.parquet"
    if features_path.exists():
        sample_df = pd.read_parquet(features_path).head(3)
        preds = predict(sample_df)

        for idx, res in enumerate(preds, 1):
            print(f"\n[Sample {idx}]")
            print(f"  Risk Score        : {res['risk_score']} / 100")
            print(f"  Threat Level      : {res['threat_level']}")
            print(f"  Attack Probability: {res['attack_probability']:.4f}")
            print(f"  Anomaly Score     : {res['anomaly_score']:.4f}")
            print(f"  Alert Triggered   : {res['alert']}")
            print(f"  Incident Priority : {res['priority']}")
            print(f"  Predicted Type    : {res['predicted_attack_type']}")
            print("  Top Indicators    :")
            for ind in res["top_indicators"]:
                print(f"    * {ind}")
    else:
        print(f"Features file not found at {features_path}.")
