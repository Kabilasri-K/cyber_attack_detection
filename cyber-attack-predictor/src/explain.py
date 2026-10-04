"""
explain.py
==========
SHAP-based Explainability Module for Cyber Attack Detection.

Translates complex tree ensemble feature attribution values into
human-readable, analyst-ready natural language explanations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# Path Configurations
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_attack_classifier.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.joblib"

# ---------------------------------------------------------------------------
# Natural Language Explanation Templates
# ---------------------------------------------------------------------------

FEATURE_TEMPLATES: Dict[str, str] = {
    "pkt_rate": (
        "Unusually high packet rate ({val:,.1f} pkts/s), indicating volumetric flood or DoS activity."
    ),
    "byte_rate": (
        "Elevated byte transmission rate ({val:,.1f} bytes/s), suggesting heavy outbound transfer or bandwidth saturation."
    ),
    "fwd_bwd_pkt_ratio": (
        "Forward-to-backward packet ratio of {val:.2f} reveals high directional asymmetry, typical of unanswered scanning probes."
    ),
    "fwd_pkt_rate": (
        "Intense forward packet transmission rate ({val:,.1f} pkts/s) without proportional response traffic."
    ),
    "bwd_pkt_rate": (
        "Anomalous backward response rate ({val:,.1f} pkts/s) signaling asymmetrical connection behavior."
    ),
    "fwd_pkt_ratio": (
        "Disproportionate forward packet share ({val:.1%}), characteristic of one-way attack probes."
    ),
    "bwd_pkt_ratio": (
        "Unusual backward packet distribution ({val:.1%}), deviating from typical bi-directional sessions."
    ),
    "syn_flag_ratio": (
        "Elevated SYN flag ratio ({val:.1%}), pointing to incomplete TCP handshakes or SYN flood behavior."
    ),
    "ack_flag_ratio": (
        "Abnormal ACK flag proportion ({val:.1%}), frequently observed in state-table exhaustion attacks."
    ),
    "rst_flag_ratio": (
        "Spike in TCP Reset (RST) flag density ({val:.1%}), resulting from closed-port rejections during port scanning."
    ),
    "fin_flag_ratio": (
        "Uncommon FIN flag frequency ({val:.1%}), characteristic of stealth reconnaissance scans."
    ),
    "psh_flag_ratio": (
        "High PSH flag activity ({val:.1%}), forcing immediate buffer pushes often seen in payload execution."
    ),
    "urg_flag_ratio": (
        "Anomalous Urgent (URG) flag count ({val:.1%}), rare in benign traffic and associated with evasion attempts."
    ),
    "syn_to_ack_ratio": (
        "Severe disparity between SYN and ACK packets (ratio {val:.2f}), signaling handshake starvation."
    ),
    "avg_pkt_size": (
        "Average packet size of {val:.1f} bytes strongly deviates from normal application traffic payloads."
    ),
    "avg_fwd_pkt_size": (
        "Forward packet payload size ({val:.1f} bytes) indicates customized attack buffers or scanning headers."
    ),
    "avg_bwd_pkt_size": (
        "Backward packet size ({val:.1f} bytes) reflects anomalous server response dimensions."
    ),
    "flow_duration_bucket": (
        "Flow duration bucket (tier {val:.0f}) corresponds to rapid automated attack execution."
    ),
    "log_flow_duration": (
        "Unusual flow longevity profile ({val:.2f}), deviating from normal session duration baselines."
    ),
}


def _format_sentence(feature_name: str, val: float) -> str:
    """Format a single feature attribution into a plain-English explanation."""
    clean_name = feature_name.strip()
    if clean_name in FEATURE_TEMPLATES:
        try:
            return FEATURE_TEMPLATES[clean_name].format(val=val)
        except Exception:
            pass

    # Humanize generic column name if not in explicit dictionary
    readable_name = clean_name.replace("_", " ").title()
    return f"Feature '{readable_name}' reached an anomalous value of {val:,.2f}, significantly driving this threat prediction."


# ---------------------------------------------------------------------------
# SHAP Explainer Wrapper
# ---------------------------------------------------------------------------

class AttackExplainer:
    """Singleton-style wrapper for fast SHAP TreeExplainer inference."""

    _instance: Optional[AttackExplainer] = None

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        self.model = joblib.load(model_path)
        self.metadata = joblib.load(metadata_path) if metadata_path.exists() else {}
        self.feature_names: List[str] = self.metadata.get("feature_names", [])

        # Initialize TreeExplainer
        self.explainer = shap.TreeExplainer(self.model)

    @classmethod
    def get_instance(cls) -> AttackExplainer:
        """Retrieve or initialize the cached explainer instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def explain(
        self,
        df: pd.DataFrame,
        top_k: int = 3,
    ) -> List[List[str]]:
        """Explain predictions for a batch of feature records.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix (columns matching model feature names).
        top_k : int, default=3
            Number of top driving features to extract per instance.

        Returns
        -------
        list of list of str
            Plain-English sentences for each record explaining the top driving features.
        """
        # Ensure correct column ordering
        if self.feature_names:
            missing_cols = [c for c in self.feature_names if c not in df.columns]
            if missing_cols:
                # Fill missing columns with 0.0
                df = df.copy()
                for c in missing_cols:
                    df[c] = 0.0
            X = df[self.feature_names].copy()
        else:
            X = df.copy()

        # Compute SHAP values
        shap_explanation = self.explainer(X)
        shap_vals = shap_explanation.values

        # Determine predicted classes
        preds = self.model.predict(X)

        all_explanations: List[List[str]] = []

        for i in range(len(X)):
            row_features = X.iloc[i]
            pred_class = int(preds[i])

            # Extract SHAP array for this instance
            # Handle multi-class (shape: n_samples, n_features, n_classes)
            if shap_vals.ndim == 3:
                # If predicted class is BENIGN (0), look at attack classes (e.g. 1 or max attack prob)
                target_c = pred_class if pred_class > 0 else 1
                if target_c >= shap_vals.shape[2]:
                    target_c = 0
                sample_shap = shap_vals[i, :, target_c]
            else:
                sample_shap = shap_vals[i]

            # Rank features by positive contribution (or highest magnitude)
            # Features that push the score highest towards attack
            ranked_indices = np.argsort(-np.abs(sample_shap))[:top_k]

            sentences: List[str] = []
            for feat_idx in ranked_indices:
                feat_name = X.columns[feat_idx]
                feat_val = float(row_features[feat_name])
                sentence = _format_sentence(feat_name, feat_val)
                sentences.append(sentence)

            all_explanations.append(sentences)

        return all_explanations


# ---------------------------------------------------------------------------
# Public Helper Functions
# ---------------------------------------------------------------------------

def explain_instance(
    row: Union[pd.Series, Dict[str, Any]],
    top_k: int = 3,
) -> List[str]:
    """Return top-K plain-English feature explanation sentences for a single record."""
    if isinstance(row, dict):
        df_row = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        df_row = pd.DataFrame([row.to_dict()])
    else:
        df_row = pd.DataFrame(row)

    explainer = AttackExplainer.get_instance()
    results = explainer.explain(df_row, top_k=top_k)
    return results[0] if results else []


def explain_prediction(
    df: pd.DataFrame,
    top_k: int = 3,
) -> List[List[str]]:
    """Return top-K plain-English feature explanations for each record in a DataFrame."""
    explainer = AttackExplainer.get_instance()
    return explainer.explain(df, top_k=top_k)
