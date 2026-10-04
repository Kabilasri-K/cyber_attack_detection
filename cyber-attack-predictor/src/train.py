"""
train.py
========
Model training and evaluation pipeline for the Cyber Attack Prediction & Early Warning System.

Models Trained:
1. XGBoost Multi-Class Classifier:
   - Predicts specific attack categories (BENIGN, DoS, PortScan, etc.).
   - Employs balanced class weighting (via sample_weight) to mitigate severe dataset class imbalance.
2. Isolation Forest Unsupervised Anomaly Detector:
   - Trained SOLELY on normal (BENIGN) network traffic baseline.
   - Detects novel, zero-day anomalous behaviors during inference.

Key Methodological Guarantees:
- Stratified Train/Test Split: Preserves exact class proportions between train and test sets,
  preventing rare attack categories from vanishing from test evaluation.
- Prevention of Data Leakage: All splits occur prior to scaling or fitting; the Isolation
  Forest strictly observes benign training data without exposure to test distributions.
- Comprehensive Evaluation: Precision, Recall, F1 (per-class and macro/weighted), ROC-AUC,
  and Confusion Matrices with plots exported to results/.
- Model Persistence: Serializes trained artifacts with metadata via joblib into models/.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Headless backend: generates plots without requiring a GUI window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# Force UTF-8 output on Windows terminals to prevent encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path Configurations (Relative to project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

LABEL_COLUMNS = ["Label", "is_attack", "attack_type"]


# ---------------------------------------------------------------------------
# Data Loading & Stratified Splitting
# ---------------------------------------------------------------------------

def load_and_split_data(
    filepath: Path = FEATURES_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, List[str], Dict[int, str]]:
    """Load feature matrix and perform stratified train/test split.

    Why Stratification is Critical:
    --------------------------------
    Network intrusion datasets (like CIC-IDS2017) are heavily imbalanced. Benign
    traffic dominates the sample space, while specific attack categories (such as
    Heartbleed, Botnet, or Web Attacks) represent small fractions of the data.
    A standard random split risks:
      1. Under-sampling or completely omitting rare attack classes in the test partition.
      2. Generating overly optimistic or high-variance performance metrics.
    Stratified splitting enforces that the distribution of each attack category
    in both training and test subsets strictly matches the overall dataset proportion.

    Data Leakage Prevention:
    ------------------------
    - Feature transformations and model fits are isolated strictly within the training set.
    - Test data is quarantined and only used for post-training evaluation.

    Assumptions:
    ------------
    - Ground truth labels in data/processed/features.parquet are assumed accurate.
    - Features provide sufficient discriminatory variance across normal and malicious traffic.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Features file not found at: {filepath}\n"
            "Run 'python src/features.py' first to build data/processed/features.parquet."
        )

    print(f"\n[1/5] Loading data from {filepath.name} ...")
    df = pd.read_parquet(filepath)

    # Separate feature matrix X from target labels
    feature_cols = [col for col in df.columns if col not in LABEL_COLUMNS]
    X = df[feature_cols].copy()

    # Targets: Multi-class (attack_type) and Binary (is_attack)
    y_multi = df["attack_type"].astype(int)
    y_binary = df["is_attack"].astype(int)

    # Build mapping from numerical attack_type to textual Label
    label_mapping: Dict[int, str] = (
        df[["attack_type", "Label"]]
        .drop_duplicates()
        .sort_values("attack_type")
        .set_index("attack_type")["Label"]
        .to_dict()
    )

    print(f"      Total records  : {len(df):,}")
    print(f"      Feature count  : {len(feature_cols)}")
    print(f"      Classes ({len(label_mapping)}): {label_mapping}")

    # Stratify by multi-class attack_type so every category is preserved proportionally
    (
        X_train,
        X_test,
        y_train_multi,
        y_test_multi,
        y_train_bin,
        y_test_bin,
    ) = train_test_split(
        X,
        y_multi,
        y_binary,
        test_size=test_size,
        random_state=random_state,
        stratify=y_multi,
    )

    print(f"      Train set size : {len(X_train):,} ({1 - test_size:.0%})")
    print(f"      Test set size  : {len(X_test):,} ({test_size:.0%})")

    return (
        X_train,
        X_test,
        y_train_multi,
        y_test_multi,
        y_train_bin,
        y_test_bin,
        feature_cols,
        label_mapping,
    )


# ---------------------------------------------------------------------------
# Model 1: XGBoost Multi-Class Classifier
# ---------------------------------------------------------------------------

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> XGBClassifier:
    """Train an XGBoost multi-class classifier using balanced sample weights.

    Class Imbalance Handling:
    -------------------------
    In multi-class settings, XGBoost does not support a single `scale_pos_weight`.
    Instead, we compute balanced sample weights using `compute_sample_weight('balanced', y_train)`.
    This inversely weights each sample by its class frequency:
        weight = n_samples / (n_classes * n_samples_in_class)
    Minority attacks receive proportionally higher gradient penalties for misclassifications.
    """
    print("\n[2/5] Training XGBoost Multi-Class Classifier ...")

    num_classes = len(np.unique(y_train))
    sample_weights = compute_sample_weight("balanced", y_train)

    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )

    xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
    print("      XGBoost training completed successfully.")
    return xgb_model


# ---------------------------------------------------------------------------
# Model 2: Isolation Forest Anomaly Detector (Trained ONLY on Normal Traffic)
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X_train: pd.DataFrame,
    y_train_binary: pd.Series,
    random_state: int = 42,
) -> IsolationForest:
    """Train an Isolation Forest exclusively on normal (BENIGN) traffic.

    Methodological Context (Zero-Day Anomaly Detection):
    ---------------------------------------------------
    - Assumption: Benign network traffic occupies specific compact topological manifolds
      in the feature space. Zero-day attacks or stealth intrusions will exhibit atypical
      structural traits (e.g. abnormal packet size variance, irregular IAT distributions).
    - Isolation Forest isolates anomalies by randomly partitioning feature values.
      Anomalous points require fewer splits (shorter path lengths) to isolate.
    - Zero Data Leakage: The model is fitted EXCLUSIVELY on training benign samples
      (is_attack == 0). It NEVER sees test traffic or attack signatures during training.
    """
    print("\n[3/5] Training Isolation Forest Anomaly Detector (Benign Baseline ONLY) ...")

    # Filter to benign training records only
    normal_mask = (y_train_binary == 0)
    X_train_normal = X_train[normal_mask].copy()

    print(f"      Fitting Isolation Forest on {len(X_train_normal):,} normal baseline samples ...")

    # Contamination parameter: expected proportion of residual noise in baseline
    iso_model = IsolationForest(
        n_estimators=150,
        max_samples="auto",
        contamination=0.02,  # Baseline assumed ~98% clean
        random_state=random_state,
        n_jobs=-1,
    )

    iso_model.fit(X_train_normal)
    print("      Isolation Forest baseline model established.")
    return iso_model


# ---------------------------------------------------------------------------
# Evaluation & Visualizations
# ---------------------------------------------------------------------------

def evaluate_xgboost(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_mapping: Dict[int, str],
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Evaluate XGBoost classifier across Precision, Recall, F1 (per-class), ROC-AUC, and Confusion Matrix."""
    print("\n[4/5] Evaluating XGBoost Classifier ...")
    results_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]

    # Metrics
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # Multi-class ROC-AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="weighted",
        )
    except Exception:
        roc_auc = 0.0

    print(f"      Accuracy / F1 (Weighted) : {f1_weighted:.4f}")
    print(f"      Macro F1                 : {f1_macro:.4f}")
    print(f"      Macro Precision          : {prec_macro:.4f}")
    print(f"      Macro Recall             : {rec_macro:.4f}")
    print(f"      ROC-AUC (Weighted OvR)   : {roc_auc:.4f}")

    # 1. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("XGBoost Multi-Class Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.tight_layout()
    cm_path = results_dir / "xgboost_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"      Saved confusion matrix -> {cm_path.name}")

    # 2. Multi-Class ROC Curves
    plt.figure(figsize=(8, 6))
    y_test_binarized = label_binarize(y_test, classes=sorted(label_mapping.keys()))
    for idx, c_name in enumerate(class_names):
        if y_test_binarized.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_test_binarized[:, idx], y_proba[:, idx])
            class_auc = roc_auc_score(y_test_binarized[:, idx], y_proba[:, idx])
            plt.plot(fpr, tpr, lw=2, label=f"{c_name} (AUC = {class_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("XGBoost Multi-Class ROC Curves (OvR)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = results_dir / "xgboost_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"      Saved ROC curve -> {roc_path.name}")

    # Per-class metrics dictionary
    report = classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "report": report,
    }


def evaluate_isolation_forest(
    model: IsolationForest,
    X_test: pd.DataFrame,
    y_test_binary: pd.Series,
    results_dir: Path = RESULTS_DIR,
) -> Dict[str, Any]:
    """Evaluate Isolation Forest anomaly detector on binary attack detection (Normal vs Anomaly)."""
    print("\n[5/5] Evaluating Isolation Forest Anomaly Detector ...")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Scikit-learn IsolationForest output: 1 for inlier (normal), -1 for outlier (anomaly)
    raw_preds = model.predict(X_test)
    y_pred_binary = (raw_preds == -1).astype(int)

    # Anomaly scores: Higher value = more anomalous
    # score_samples returns negative anomaly score (lower is more anomalous)
    anomaly_scores = -model.score_samples(X_test)

    prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)
    roc_auc = roc_auc_score(y_test_binary, anomaly_scores)

    print(f"      Binary Precision (Attack): {prec:.4f}")
    print(f"      Binary Recall (Attack)   : {rec:.4f}")
    print(f"      Binary F1-Score          : {f1:.4f}")
    print(f"      ROC-AUC Score            : {roc_auc:.4f}")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=["Normal (Benign)", "Attack (Anomaly)"],
        yticklabels=["Normal (Benign)", "Attack (Anomaly)"],
    )
    plt.title("Isolation Forest Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Ground Truth", fontsize=11)
    plt.tight_layout()
    cm_path = results_dir / "isolation_forest_confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"      Saved confusion matrix -> {cm_path.name}")

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test_binary, anomaly_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="forestgreen", lw=2, label=f"Isolation Forest (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("Isolation Forest ROC Curve", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = results_dir / "isolation_forest_roc_curve.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"      Saved ROC curve -> {roc_path.name}")

    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Summary Table Display & Serialization
# ---------------------------------------------------------------------------

def display_summary(
    xgb_metrics: Dict[str, Any],
    iso_metrics: Dict[str, Any],
    label_mapping: Dict[int, str],
) -> None:
    """Print a clean, structured summary table of model performance metrics."""
    print("\n" + "=" * 78)
    print("              MODEL PERFORMANCE & EVALUATION SUMMARY TABLE              ")
    print("=" * 78)

    print(f"{'Model':<22} | {'Evaluation Task':<20} | {'Prec':<7} | {'Rec':<7} | {'F1':<7} | {'ROC-AUC':<7}")
    print("-" * 78)

    # XGBoost row
    print(
        f"{'XGBoost Classifier':<22} | "
        f"{'Multi-Class (Macro)':<20} | "
        f"{xgb_metrics['precision_macro']:<7.4f} | "
        f"{xgb_metrics['recall_macro']:<7.4f} | "
        f"{xgb_metrics['f1_macro']:<7.4f} | "
        f"{xgb_metrics['roc_auc']:<7.4f}"
    )

    # Isolation Forest row
    print(
        f"{'Isolation Forest':<22} | "
        f"{'Anomaly (Binary)':<20} | "
        f"{iso_metrics['precision']:<7.4f} | "
        f"{iso_metrics['recall']:<7.4f} | "
        f"{iso_metrics['f1']:<7.4f} | "
        f"{iso_metrics['roc_auc']:<7.4f}"
    )
    print("=" * 78)

    print("\n[Per-Class Breakdown for XGBoost]")
    print(f"{'Class Name':<16} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 62)
    report = xgb_metrics["report"]
    for idx, c_name in label_mapping.items():
        if c_name in report:
            row = report[c_name]
            print(f"{c_name:<16} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f1-score']:<10.4f} | {int(row['support']):<8}")
    print("=" * 62 + "\n")


def save_models(
    xgb_model: XGBClassifier,
    iso_model: IsolationForest,
    feature_names: List[str],
    label_mapping: Dict[int, str],
    xgb_metrics: Dict[str, Any],
    iso_metrics: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
) -> None:
    """Save trained models and operational metadata using joblib."""
    models_dir.mkdir(parents=True, exist_ok=True)

    xgb_path = models_dir / "xgboost_attack_classifier.joblib"
    iso_path = models_dir / "isolation_forest_detector.joblib"
    meta_path = models_dir / "model_metadata.joblib"

    print(f"Saving serialized models to {models_dir.resolve()} ...")

    # 1. XGBoost model
    joblib.dump(xgb_model, xgb_path)
    print(f"  [OK] Saved XGBoost classifier      -> {xgb_path.name}")

    # 2. Isolation Forest model
    joblib.dump(iso_model, iso_path)
    print(f"  [OK] Saved Isolation Forest model  -> {iso_path.name}")

    # 3. Comprehensive Pipeline Metadata
    metadata = {
        "feature_names": feature_names,
        "label_mapping": label_mapping,
        "inverse_label_mapping": {v: k for k, v in label_mapping.items()},
        "xgb_metrics": {
            "f1_macro": xgb_metrics["f1_macro"],
            "f1_weighted": xgb_metrics["f1_weighted"],
            "roc_auc": xgb_metrics["roc_auc"],
        },
        "iso_metrics": {
            "f1": iso_metrics["f1"],
            "roc_auc": iso_metrics["roc_auc"],
        },
    }
    joblib.dump(metadata, meta_path)
    print(f"  [OK] Saved pipeline metadata       -> {meta_path.name}")


# ---------------------------------------------------------------------------
# Main Training Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Execute end-to-end training and evaluation pipeline."""
    print("=" * 78)
    print("   CYBER ATTACK PREDICTION & EARLY WARNING SYSTEM — MODEL TRAINING PIPELINE")
    print("=" * 78)

    # Step 1: Load and Stratify
    (
        X_train,
        X_test,
        y_train_multi,
        y_test_multi,
        y_train_bin,
        y_test_bin,
        feature_names,
        label_mapping,
    ) = load_and_split_data()

    # Step 2: Train XGBoost Multi-Class Classifier with Class Weights
    xgb_model = train_xgboost(X_train, y_train_multi)

    # Step 3: Train Isolation Forest on Normal Traffic Only
    iso_model = train_isolation_forest(X_train, y_train_bin)

    # Step 4: Evaluate Models
    xgb_metrics = evaluate_xgboost(xgb_model, X_test, y_test_multi, label_mapping)
    iso_metrics = evaluate_isolation_forest(iso_model, X_test, y_test_bin)

    # Step 5: Display Summary Table
    display_summary(xgb_metrics, iso_metrics, label_mapping)

    # Step 6: Serialize Models & Artifacts
    save_models(
        xgb_model,
        iso_model,
        feature_names,
        label_mapping,
        xgb_metrics,
        iso_metrics,
    )

    print("[COMPLETE] Training, evaluation, plotting, and serialization completed successfully.\n")


if __name__ == "__main__":
    main()
