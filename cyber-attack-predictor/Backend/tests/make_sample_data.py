"""
make_sample_data.py — creates a tiny synthetic CIC-IDS2017-style CSV in
data/raw/ so that preprocessing.py can be exercised without the real dataset.
Delete this file (and the generated CSVs) before using the real data.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N = 2000  # rows per file

LABELS = [
    "BENIGN", "BENIGN", "BENIGN",          # 3-in-5 chance benign
    "DoS Hulk", "PortScan",
]

def make_chunk(n: int, label_pool: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        " Flow Duration": rng.integers(0, 1_000_000, n),          # leading space on purpose
        "Total Fwd Packets": rng.integers(0, 500, n),
        "Total Backward Packets": rng.integers(0, 500, n),
        "Flow Bytes/s": rng.uniform(0, 1e7, n),
        "Flow Packets/s": rng.uniform(0, 1e5, n),
        "Fwd PSH Flags": rng.integers(0, 2, n),
        "Bwd PSH Flags": rng.integers(0, 2, n),
        "Constant Col": 0,                                          # constant col
        "Label": rng.choice(label_pool, n),
    })
    # Inject some inf values
    df.loc[rng.integers(0, n, 10), "Flow Bytes/s"] = np.inf
    df.loc[rng.integers(0, n, 5),  "Flow Packets/s"] = -np.inf
    # Inject some NaN values
    df.loc[rng.integers(0, n, 8), "Total Fwd Packets"] = np.nan
    # Inject exact duplicates
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)
    return df

for i, fname in enumerate(["Monday-WorkingHours.pcap_ISCX.csv",
                             "Tuesday-WorkingHours.pcap_ISCX.csv"], 1):
    chunk = make_chunk(N, LABELS)
    out_path = OUT_DIR / fname
    chunk.to_csv(out_path, index=False)
    print(f"Created {out_path.name}  ({len(chunk)} rows)")

print("Done.")
