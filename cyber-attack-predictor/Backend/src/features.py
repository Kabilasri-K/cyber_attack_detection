"""
features.py
===========
Engineers behavioural features from the cleaned CIC-IDS2017 dataset
(data/processed/clean.parquet) and saves the enriched feature matrix to
data/processed/features.parquet.

Design principles
-----------------
* Every feature has an explanatory comment describing its cybersecurity significance.
* All divide-by-zero cases are safely handled with safe_div() (returns 0.0 or specified fill value).
* Seamlessly handles both real CIC-IDS2017 CSV data (~80 features) and smaller/synthetic subsets.
* Returns (feature_df, feature_names) where feature_names contains all engineered feature columns.
* Saves results to data/processed/features.parquet.

Run directly:
    python src/features.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Force UTF-8 output on Windows terminals (prevents encoding issues with special characters)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Path helpers (relative to project root — no hardcoded absolute paths)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

# Label columns to preserve for model training but exclude from feature names
LABEL_COLS = {"Label", "is_attack", "attack_type"}


# ---------------------------------------------------------------------------
# Safe division utility
# ---------------------------------------------------------------------------

def safe_div(
    numerator: pd.Series | float | np.ndarray,
    denominator: pd.Series | float | np.ndarray,
    fill_value: float = 0.0,
) -> pd.Series:
    """Safely divide numerator by denominator element-wise, replacing zero or invalid denominators.

    Parameters
    ----------
    numerator : pd.Series or array-like
        The dividend values.
    denominator : pd.Series or array-like
        The divisor values. Zeroes, NaNs, and infinities are replaced with NaN before division.
    fill_value : float, default=0.0
        Value substituted where division by zero or NaN occurs.

    Returns
    -------
    pd.Series
        Cleaned division result without inf or NaN.
    """
    num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    den = pd.Series(denominator) if not isinstance(denominator, pd.Series) else denominator

    # Replace zero, inf, -inf in denominator with NaN to avoid division by zero warning
    safe_den = den.replace([0, 0.0, np.inf, -np.inf], np.nan)
    result = num / safe_den
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def _get_series(df: pd.DataFrame, col_names: list[str], default_val: float = 0.0) -> pd.Series:
    """Retrieve the first matching column from col_names in df, or return a default Series."""
    for col in col_names:
        if col in df.columns:
            return df[col].astype(float)
    return pd.Series(default_val, index=df.index, dtype=float)


# ---------------------------------------------------------------------------
# Feature Engineering Modules
# ---------------------------------------------------------------------------

def compute_rate_features(df: pd.DataFrame, out: dict[str, pd.Series]) -> None:
    """Compute packet and byte rates per second.

    Cybersecurity context:
    - Attackers performing DoS/DDoS flood target networks with abnormal volumes
      of packets or bytes per second to exhaust bandwidth or server resources.
    - Port scans generate brief bursts of packets at varying rates.
    """
    # Duration in seconds (CIC-IDS2017 stores Flow Duration in microseconds)
    flow_dur_sec = _get_series(df, ["Flow Duration"]) / 1_000_000.0

    fwd_pkts = _get_series(df, ["Total Fwd Packets"])
    bwd_pkts = _get_series(df, ["Total Backward Packets"])
    total_pkts = fwd_pkts + bwd_pkts

    # 1. Packet rate (packets/sec)
    # Explanation: Abnormal surge in total packets per second indicates active flood or DDoS attack.
    if "Flow Packets/s" in df.columns:
        out["pkt_rate"] = df["Flow Packets/s"].clip(lower=0).fillna(0.0)
    else:
        out["pkt_rate"] = safe_div(total_pkts, flow_dur_sec, fill_value=0.0)

    # 2. Byte rate (bytes/sec)
    # Explanation: High throughput indicates volumetric attacks (e.g. UDP/ICMP floods) or bulk data exfiltration.
    fwd_bytes = _get_series(df, ["Total Length of Fwd Packets"])
    bwd_bytes = _get_series(df, ["Total Length of Bwd Packets"])
    total_bytes = fwd_bytes + bwd_bytes

    if "Flow Bytes/s" in df.columns:
        out["byte_rate"] = df["Flow Bytes/s"].clip(lower=0).fillna(0.0)
    elif "Total Length of Fwd Packets" in df.columns or "Total Length of Bwd Packets" in df.columns:
        out["byte_rate"] = safe_div(total_bytes, flow_dur_sec, fill_value=0.0)
    else:
        out["byte_rate"] = safe_div(total_pkts * 64.0, flow_dur_sec, fill_value=0.0)

    # 3. Forward packet rate (fwd packets/sec)
    # Explanation: High forward packet rate with negligible response indicates unilateral traffic (e.g., SYN flood).
    out["fwd_pkt_rate"] = safe_div(fwd_pkts, flow_dur_sec, fill_value=0.0)

    # 4. Backward packet rate (bwd packets/sec)
    # Explanation: Low or zero backward rate indicates the server is not replying or dropping packets.
    out["bwd_pkt_rate"] = safe_div(bwd_pkts, flow_dur_sec, fill_value=0.0)


def compute_directional_features(df: pd.DataFrame, out: dict[str, pd.Series]) -> None:
    """Compute directional flow ratios between forward and backward traffic.

    Cybersecurity context:
    - Normal benign TCP flows are conversational: requests elicit responses (ratio close to balanced).
    - Port scans and SYN floods send forward packets without waiting for or receiving symmetric responses.
    - Exfiltration flows exhibit high forward-to-backward byte ratios.
    """
    fwd_pkts = _get_series(df, ["Total Fwd Packets"])
    bwd_pkts = _get_series(df, ["Total Backward Packets"])
    total_pkts = fwd_pkts + bwd_pkts

    # 5. Ratio of forward packets to backward packets (asymmetry ratio)
    # Explanation: Large values (>> 1) indicate unilateral sender bursts typical of port sweeps and SYN floods.
    out["fwd_bwd_pkt_ratio"] = safe_div(fwd_pkts, bwd_pkts, fill_value=0.0)

    # 6. Forward packet ratio (fwd packets / total packets)
    # Explanation: Value close to 1.0 implies outgoing packets without receiving replies; close to 0.0 implies ingress flood.
    out["fwd_pkt_ratio"] = safe_div(fwd_pkts, total_pkts, fill_value=0.0)

    # 7. Backward packet ratio (bwd packets / total packets)
    # Explanation: Measures response proportion; complementary to fwd_pkt_ratio.
    out["bwd_pkt_ratio"] = safe_div(bwd_pkts, total_pkts, fill_value=0.0)

    # 8. Byte directional ratios
    fwd_bytes = _get_series(df, ["Total Length of Fwd Packets"])
    bwd_bytes = _get_series(df, ["Total Length of Bwd Packets"])
    total_bytes = fwd_bytes + bwd_bytes

    if "Total Length of Fwd Packets" in df.columns or "Total Length of Bwd Packets" in df.columns:
        # Explanation: High ratio of forward bytes indicates upload/exfiltration; high backward bytes indicates download.
        out["fwd_bwd_byte_ratio"] = safe_div(fwd_bytes, bwd_bytes, fill_value=0.0)
        out["fwd_byte_ratio"] = safe_div(fwd_bytes, total_bytes, fill_value=0.0)
    elif "Down/Up Ratio" in df.columns:
        out["fwd_bwd_byte_ratio"] = safe_div(1.0, df["Down/Up Ratio"], fill_value=0.0)
        out["fwd_byte_ratio"] = safe_div(1.0, 1.0 + df["Down/Up Ratio"], fill_value=0.5)


def compute_tcp_flag_features(df: pd.DataFrame, out: dict[str, pd.Series]) -> None:
    """Compute TCP control flag ratios normalized by total packets.

    Cybersecurity context:
    - SYN flood: disproportionate number of SYN packets without subsequent ACK packets.
    - Port scans (Nmap/masscan): produce high RST flags (closed ports) or stealth flags (FIN, NULL, XMAS).
    - ACK floods: high ratio of ACK packets sent without prior handshakes to overwhelm stateful firewalls.
    """
    fwd_pkts = _get_series(df, ["Total Fwd Packets"])
    bwd_pkts = _get_series(df, ["Total Backward Packets"])
    total_pkts = fwd_pkts + bwd_pkts
    # If total_pkts is all zeros, fall back to 1.0 to avoid zero division
    denom = total_pkts.replace(0, 1.0)

    # 9. SYN flag ratio
    # Explanation: SYN packets initiate connections; a high ratio without ACKs indicates a SYN flood attack.
    syn_flags = _get_series(df, ["SYN Flag Count", "Fwd PSH Flags"]) if "SYN Flag Count" in df.columns else _get_series(df, ["SYN Flag Count"])
    out["syn_flag_ratio"] = safe_div(syn_flags, denom, fill_value=0.0)

    # 10. ACK flag ratio
    # Explanation: ACK packets confirm received data; unusually high ratio can signify ACK flooding attacks.
    ack_flags = _get_series(df, ["ACK Flag Count"])
    out["ack_flag_ratio"] = safe_div(ack_flags, denom, fill_value=0.0)

    # 11. RST flag ratio
    # Explanation: RST (reset) terminates connections; port scans against closed ports trigger immediate RST packets.
    rst_flags = _get_series(df, ["RST Flag Count"])
    out["rst_flag_ratio"] = safe_div(rst_flags, denom, fill_value=0.0)

    # 12. FIN flag ratio
    # Explanation: Graceful teardown flag; anomalous counts appear in stealth port scans (e.g. FIN scan).
    fin_flags = _get_series(df, ["FIN Flag Count"])
    out["fin_flag_ratio"] = safe_div(fin_flags, denom, fill_value=0.0)

    # 13. PSH flag ratio
    # Explanation: Push flag forces buffer flush; frequent PSH flags appear in interactive shell sessions or HTTP attacks.
    psh_flags = _get_series(df, ["PSH Flag Count", "Fwd PSH Flags"])
    if "Bwd PSH Flags" in df.columns:
        psh_flags = psh_flags + df["Bwd PSH Flags"].fillna(0.0)
    out["psh_flag_ratio"] = safe_div(psh_flags, denom, fill_value=0.0)

    # 14. URG flag ratio
    # Explanation: Urgent flag is rarely used in standard web traffic; elevated levels indicate evasion or malformed packets.
    urg_flags = _get_series(df, ["URG Flag Count", "Fwd URG Flags"])
    out["urg_flag_ratio"] = safe_div(urg_flags, denom, fill_value=0.0)

    # 15. SYN to ACK ratio
    # Explanation: Severe divergence (SYN >> ACK) indicates incomplete handshakes and state exhaustion attacks.
    out["syn_to_ack_ratio"] = safe_div(syn_flags, ack_flags, fill_value=0.0)


def compute_packet_size_features(df: pd.DataFrame, out: dict[str, pd.Series]) -> None:
    """Compute average packet sizes and segment statistics.

    Cybersecurity context:
    - Port scans use minimal packet sizes (~40-60 bytes, headers only).
    - Buffer overflow and volumetric DoS attacks use maximum transmission unit (MTU) packet sizes (~1400-1500 bytes).
    - Normal browsing features high variance (small ACKs + large payload packets).
    """
    fwd_pkts = _get_series(df, ["Total Fwd Packets"])
    bwd_pkts = _get_series(df, ["Total Backward Packets"])
    total_pkts = fwd_pkts + bwd_pkts

    fwd_bytes = _get_series(df, ["Total Length of Fwd Packets"])
    bwd_bytes = _get_series(df, ["Total Length of Bwd Packets"])
    total_bytes = fwd_bytes + bwd_bytes

    # 16. Average overall packet size
    # Explanation: Differentiates tiny probe packets (scans) from heavy payload data transfers.
    if "Average Packet Size" in df.columns:
        out["avg_pkt_size"] = df["Average Packet Size"].clip(lower=0).fillna(0.0)
    elif "Packet Length Mean" in df.columns:
        out["avg_pkt_size"] = df["Packet Length Mean"].clip(lower=0).fillna(0.0)
    elif "Total Length of Fwd Packets" in df.columns and "Total Length of Bwd Packets" in df.columns:
        out["avg_pkt_size"] = safe_div(total_bytes, total_pkts, fill_value=0.0)
    elif "Flow Bytes/s" in df.columns and "Flow Packets/s" in df.columns:
        out["avg_pkt_size"] = safe_div(df["Flow Bytes/s"], df["Flow Packets/s"], fill_value=0.0)
    else:
        out["avg_pkt_size"] = pd.Series(0.0, index=df.index)

    # 17. Average forward packet size
    # Explanation: Small forward size implies pure control packets (probes); large forward size indicates data upload.
    if "Avg Fwd Segment Size" in df.columns:
        out["avg_fwd_pkt_size"] = df["Avg Fwd Segment Size"].clip(lower=0).fillna(0.0)
    else:
        out["avg_fwd_pkt_size"] = safe_div(fwd_bytes, fwd_pkts, fill_value=0.0)

    # 18. Average backward packet size
    # Explanation: Measures response payload dimension; small backward size suggests error responses or RSTs.
    if "Avg Bwd Segment Size" in df.columns:
        out["avg_bwd_pkt_size"] = df["Avg Bwd Segment Size"].clip(lower=0).fillna(0.0)
    else:
        out["avg_bwd_pkt_size"] = safe_div(bwd_bytes, bwd_pkts, fill_value=0.0)


def compute_duration_bucket_features(df: pd.DataFrame, out: dict[str, pd.Series]) -> None:
    """Bin flow duration into ordinal behavioral categories.

    Cybersecurity context:
    - Micro-duration flows (< 100 ms) are characteristic of scanning tools (Nmap fast scan) and SYN floods.
    - Long-duration flows (> 60 s) correlate with remote access shells (RATs), C2 beaconing, or large transfers.
    """
    dur_microseconds = _get_series(df, ["Flow Duration"])

    # Buckets:
    # 0 = Very short (< 100 ms): Port scan, quick probe, rejected handshake
    # 1 = Short (100 ms - 1 s): Fast web request or short transaction
    # 2 = Medium (1 s - 10 s): Standard interactive web session
    # 3 = Long (10 s - 60 s): Streaming, sustained download, or brute-force trial
    # 4 = Very long (> 60 s): Persistent connection, C2 channel, or reverse shell
    bins = [-1.0, 100_000.0, 1_000_000.0, 10_000_000.0, 60_000_000.0, float("inf")]
    labels = [0, 1, 2, 3, 4]

    # Explanation: Categorizes flow longevity into discrete ordinal tiers for non-linear tree splits.
    out["flow_duration_bucket"] = (
        pd.cut(dur_microseconds, bins=bins, labels=labels, right=True)
        .astype(int)
    )

    # Explanation: Log-transformed flow duration to compress heavy-tailed distributions and stabilize learning.
    out["log_flow_duration"] = np.log1p(dur_microseconds.clip(lower=0))


def compute_cic_advanced_features(df: pd.DataFrame, out: dict[str, pd.Series]) -> None:
    """Extract and pass through available advanced CIC-IDS2017 flow properties.

    Cybersecurity context:
    - Inter-arrival time (IAT) statistics reveal automated periodicity (botnets, C2 intervals).
    - Window size parameters reflect OS fingerprinting and connection exhaustion.
    """
    cic_candidates = [
        # Inter-arrival times (automated attacks have ultra-low mean IAT and low std)
        "Flow IAT Mean",
        "Flow IAT Std",
        "Flow IAT Max",
        "Flow IAT Min",
        "Fwd IAT Mean",
        "Bwd IAT Mean",
        # Packet length statistics
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Max Packet Length",
        "Min Packet Length",
        # Header size & window statistics
        "Fwd Header Length",
        "Bwd Header Length",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
        # Activity vs Inactivity (slowloris attacks have high idle times)
        "Active Mean",
        "Idle Mean",
        "Down/Up Ratio",
    ]

    for col in cic_candidates:
        if col in df.columns:
            feat_name = col.lower().replace(" ", "_").replace("/", "_")
            if feat_name not in out:
                # Explanation: Direct pass-through of verified CIC-IDS2017 behavioral metric.
                out[feat_name] = df[col].clip(lower=0).fillna(0.0)


# ---------------------------------------------------------------------------
# Master Pipeline Function
# ---------------------------------------------------------------------------

def build_features(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> tuple[pd.DataFrame, list[str]]:
    """Load cleaned data, engineer all behavioral attack detection features, and save to Parquet.

    Parameters
    ----------
    input_path : Path, default=data/processed/clean.parquet
        Source cleaned dataframe file.
    output_path : Path, default=data/processed/features.parquet
        Destination parquet file path.

    Returns
    -------
    feature_df : pd.DataFrame
        Complete dataframe containing engineered features and label columns.
    feature_names : list[str]
        List of strings specifying all engineered feature column names.
    """
    print("\n" + "=" * 60)
    print("  Cyber Attack Detection — Behavioral Feature Engineering")
    print("=" * 60)

    input_file = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at: {input_file.resolve()}")

    print(f"  Reading dataset: {input_file.name} ...")
    df = pd.read_parquet(input_file)
    print(f"  Input dimensions: {df.shape[0]:,} rows x {df.shape[1]} columns")

    feature_dict: dict[str, pd.Series] = {}

    # 1. Flow rate features (packet rate, byte rate, fwd/bwd rates)
    compute_rate_features(df, feature_dict)

    # 2. Directional ratios (forward/backward balance, byte proportions)
    compute_directional_features(df, feature_dict)

    # 3. TCP flag ratios (SYN, ACK, RST, FIN, PSH, URG, and SYN/ACK ratio)
    compute_tcp_flag_features(df, feature_dict)

    # 4. Packet size metrics (average size, forward/backward average sizes)
    compute_packet_size_features(df, feature_dict)

    # 5. Flow duration buckets and log transformations
    compute_duration_bucket_features(df, feature_dict)

    # 6. Advanced CIC-IDS2017 metrics (IAT, window size, header lengths)
    compute_cic_advanced_features(df, feature_dict)

    # Assemble feature DataFrame
    feat_df = pd.DataFrame(feature_dict, index=df.index)

    # Clean any residual infinities or NaNs resulting from math operations
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Feature names list (engineered features only)
    feature_names = list(feat_df.columns)

    # Retain label columns for training/evaluation pipelines
    for label_col in LABEL_COLS:
        if label_col in df.columns:
            feat_df[label_col] = df[label_col].values

    # Output directory verification
    output_file.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(output_file, index=False, compression="snappy")
    size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"\n  Engineered features count : {len(feature_names)}")
    print(f"  Total records processed   : {len(feat_df):,}")
    print(f"  Saved feature matrix to   : {output_file.name} ({size_mb:.2f} MB)")
    print("=" * 60 + "\n")

    return feat_df, feature_names


if __name__ == "__main__":
    try:
        features_df, names = build_features()
        print(f"Features list ({len(names)} features):")
        for i, name in enumerate(names, 1):
            print(f"  {i:2d}. {name}")
    except Exception as exc:
        print(f"[ERROR] Failed during feature engineering: {exc}", file=sys.stderr)
        sys.exit(1)
