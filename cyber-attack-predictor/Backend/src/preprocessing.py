"""
preprocessing.py
================
Loads all CIC-IDS2017 CSV files from  data/raw/,
cleans the data (whitespace, inf, NaN, duplicates, constant cols),
encodes the Label column into binary (is_attack) and multi-class
(attack_type) columns, then saves the result to
data/processed/clean.parquet.

Run directly:
    python src/preprocessing.py
"""

from __future__ import annotations

import io
import sys

# Force UTF-8 output on Windows terminals (avoids cp1252 encode errors)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path helpers  (all paths relative to project root — no hardcoded absolutes)
# ---------------------------------------------------------------------------

# src/preprocessing.py → ../ = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH  = PROCESSED_DIR / "clean.parquet"


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------

def load_csvs(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and vertically concatenate every CSV file found in *raw_dir*.

    Parameters
    ----------
    raw_dir : Path
        Directory that contains the CIC-IDS2017 CSV files.

    Returns
    -------
    pd.DataFrame
        Single DataFrame combining all CSV files.

    Raises
    ------
    FileNotFoundError
        If *raw_dir* does not exist or contains no CSV files.
    """
    csv_files = sorted(raw_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{raw_dir}'.\n"
            "Download the CIC-IDS2017 dataset and place the CSV files in data/raw/.\n"
            "Download link: https://www.unb.ca/cic/datasets/ids-2017.html"
        )

    print(f"\n{'='*60}")
    print(f"  Found {len(csv_files)} CSV file(s) in {raw_dir.name}/")
    print(f"{'='*60}")

    frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        print(f"  Loading  {csv_path.name} ...", end=" ", flush=True)
        df_chunk = pd.read_csv(csv_path, low_memory=False)
        print(f"{len(df_chunk):,} rows")
        frames.append(df_chunk)

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows after concat : {len(combined):,}")
    print(f"  Total columns           : {combined.shape[1]}")
    return combined


# ---------------------------------------------------------------------------
# Step 2 — Strip whitespace from column names
# ---------------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from every column name.

    CIC-IDS2017 CSVs are notorious for columns like ' Flow Duration'
    (leading space).  This normalises them all.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Same DataFrame with sanitised column names (in-place rename).
    """
    original = list(df.columns)
    df.columns = df.columns.str.strip()
    changed = sum(o != n for o, n in zip(original, df.columns))
    print(f"\n[clean_column_names]  Fixed {changed} column name(s) with whitespace.")
    return df


# ---------------------------------------------------------------------------
# Step 3 — Replace inf / -inf with NaN
# ---------------------------------------------------------------------------

def replace_infinities(df: pd.DataFrame) -> pd.DataFrame:
    """Replace numpy inf and -inf values with NaN across all numeric columns.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols].values).sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    print(f"[replace_infinities]  Replaced {inf_count:,} inf/-inf value(s) with NaN.")
    return df


# ---------------------------------------------------------------------------
# Step 4 — Drop NaN rows and duplicate rows
# ---------------------------------------------------------------------------

def drop_nulls_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain any NaN value, then drop exact duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    rows_before = len(df)

    # Drop rows with any NaN
    df = df.dropna()
    rows_after_nan = len(df)
    print(
        f"[drop_nulls]       Removed {rows_before - rows_after_nan:,} row(s) "
        f"containing NaN  ->  {rows_after_nan:,} rows remain."
    )

    # Drop exact duplicates
    df = df.drop_duplicates()
    rows_after_dup = len(df)
    print(
        f"[drop_duplicates]  Removed {rows_after_nan - rows_after_dup:,} duplicate row(s) "
        f"->  {rows_after_dup:,} rows remain."
    )
    return df


# ---------------------------------------------------------------------------
# Step 5 — Drop constant columns
# ---------------------------------------------------------------------------

def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns whose values are all the same (zero variance).

    Constant features carry no predictive information and bloat the model.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    # nunique(dropna=False) == 1 catches both all-same and all-NaN (already removed above)
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    if constant_cols:
        df = df.drop(columns=constant_cols)
        print(
            f"[drop_constant_cols]  Dropped {len(constant_cols)} constant column(s): "
            f"{constant_cols}"
        )
    else:
        print("[drop_constant_cols]  No constant columns found.")
    return df


# ---------------------------------------------------------------------------
# Step 6 — Encode the Label column
# ---------------------------------------------------------------------------

# CIC-IDS2017 uses 'BENIGN' as the normal-traffic label.
BENIGN_LABEL = "BENIGN"


def encode_labels(df: pd.DataFrame, label_col: str = "Label") -> pd.DataFrame:
    """Create binary (is_attack) and integer multi-class (attack_type) columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column named *label_col*.
    label_col : str
        Name of the raw label column (default: 'Label').

    Returns
    -------
    pd.DataFrame
        DataFrame with two extra columns:
        - ``is_attack``   : int  0 = benign, 1 = attack
        - ``attack_type`` : int  0 = benign, 1..N = each unique attack class

    Raises
    ------
    KeyError
        If *label_col* is not found in *df*.
    """
    if label_col not in df.columns:
        raise KeyError(
            f"Column '{label_col}' not found. Available columns: {list(df.columns)}"
        )

    # Normalise label strings (strip whitespace, consistent casing)
    labels_clean = df[label_col].astype(str).str.strip()

    # --- Binary label ---
    df["is_attack"] = (labels_clean != BENIGN_LABEL).astype(int)

    # --- Multi-class label ---
    # Sort categories so BENIGN is always 0, attacks get 1..N alphabetically
    categories = sorted(labels_clean.unique(), key=lambda x: (x != BENIGN_LABEL, x))
    cat_dtype = pd.CategoricalDtype(categories=categories, ordered=False)
    df["attack_type"] = labels_clean.astype(cat_dtype).cat.codes  # int8

    # Build and print the mapping for transparency
    code_map = {cat: code for code, cat in enumerate(categories)}
    print(f"\n[encode_labels]  Label column : '{label_col}'")
    print(f"  attack_type encoding:")
    for cat, code in code_map.items():
        print(f"    {code:>3}  →  {cat}")

    return df


# ---------------------------------------------------------------------------
# Step 7 — Save to Parquet
# ---------------------------------------------------------------------------

def save_parquet(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Save the cleaned DataFrame to a Parquet file using snappy compression.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset to persist.
    output_path : Path
        Destination file path.  Parent directories are created if needed.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"\n[save_parquet]  Saved → {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Step 8 — Print class distribution
# ---------------------------------------------------------------------------

def print_class_distribution(df: pd.DataFrame, label_col: str = "Label") -> None:
    """Print the count and percentage for each label class.

    Parameters
    ----------
    df : pd.DataFrame
    label_col : str
        Raw label column (used for readable class names in the output).
    """
    total = len(df)
    print(f"\n{'='*60}")
    print("  Class distribution (attack_type)")
    print(f"{'='*60}")
    print(f"  {'Code':>5}  {'Label':<40}  {'Count':>10}  {'%':>7}")
    print(f"  {'-'*5}  {'-'*40}  {'-'*10}  {'-'*7}")

    # Group by both encoded column and original label for readability
    dist = (
        df.groupby(["attack_type", label_col], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values("attack_type")
    )
    for _, row in dist.iterrows():
        pct = 100.0 * row["count"] / total
        print(f"  {int(row['attack_type']):>5}  {row[label_col]:<40}  {int(row['count']):>10,}  {pct:>6.2f}%")

    # Binary summary
    print(f"\n  Binary split:")
    benign_n  = (df["is_attack"] == 0).sum()
    attack_n  = (df["is_attack"] == 1).sum()
    print(f"    Benign  : {benign_n:>10,}  ({100*benign_n/total:.2f}%)")
    print(f"    Attack  : {attack_n:>10,}  ({100*attack_n/total:.2f}%)")
    print(f"  Total rows: {total:,}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    raw_dir: Path = RAW_DIR,
    output_path: Path = OUTPUT_PATH,
    label_col: str = "Label",
) -> pd.DataFrame:
    """Execute the full preprocessing pipeline end-to-end.

    Steps
    -----
    1. Load all CSVs from *raw_dir*.
    2. Strip whitespace from column names.
    3. Replace inf / -inf with NaN.
    4. Drop NaN rows and duplicate rows.
    5. Drop constant columns.
    6. Encode the label column.
    7. Print class distribution.
    8. Save to *output_path* as Parquet.

    Parameters
    ----------
    raw_dir : Path
        Folder containing CIC-IDS2017 CSV files.
    output_path : Path
        Where to write the cleaned Parquet file.
    label_col : str
        Name of the label column in the CSVs.

    Returns
    -------
    pd.DataFrame
        The fully cleaned and encoded dataset.
    """
    print("\n" + "="*60)
    print("  Cyber Attack Detection — Preprocessing Pipeline")
    print("="*60)

    # --- Load ---
    df = load_csvs(raw_dir)
    rows_raw = len(df)

    # --- Clean ---
    df = clean_column_names(df)
    df = replace_infinities(df)
    df = drop_nulls_and_duplicates(df)
    df = drop_constant_columns(df)

    # --- Encode ---
    df = encode_labels(df, label_col=label_col)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Row count summary")
    print(f"{'='*60}")
    print(f"  Before cleaning : {rows_raw:>10,}")
    print(f"  After  cleaning : {len(df):>10,}")
    print(f"  Rows removed    : {rows_raw - len(df):>10,}  "
          f"({100*(rows_raw - len(df))/max(rows_raw, 1):.2f}%)")

    print_class_distribution(df, label_col=label_col)

    # --- Save ---
    save_parquet(df, output_path)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"\n[ERROR]  {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyError as exc:
        print(f"\n[ERROR]  {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR]  {exc}", file=sys.stderr)
        raise
