"""Data loading and time-based splitting for the credit card fraud dataset.

Design decisions:
- Sort by 'Time' then split positionally (70 / 15 / 15).
  This prevents data leakage from future transactions.
- No preprocessing here — scalers are fit on train only in Phase 2.

Usage:
    python -m ml.data
"""
from __future__ import annotations

import pandas as pd

from ml.config import (
    RAW_DATA_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
    ALL_FEATURES,
    TARGET,
)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV.

    Args:
        path: Optional override for the CSV file path.
              Defaults to ``config.RAW_DATA_PATH``.

    Returns:
        The raw DataFrame with all expected columns validated.
    """
    csv_path = path or RAW_DATA_PATH
    df = pd.read_csv(csv_path)

    # Basic validation
    assert TARGET in df.columns, f"Missing target column '{TARGET}'"
    for feat in ALL_FEATURES:
        assert feat in df.columns, f"Missing feature column '{feat}'"
    assert len(df) > 0, "Dataset is empty"

    return df


def time_based_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset chronologically into train / val / test.

    The 'Time' column represents seconds elapsed from the first
    transaction in the dataset (~48 hours total).  We sort by Time
    and take the earliest 70 % as train, next 15 % as validation,
    and last 15 % as test.

    This mirrors production where the model only sees past data.

    Returns:
        ``(train, val, test)`` DataFrames.
    """
    df_sorted = df.sort_values("Time").reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df_sorted.iloc[:train_end].copy()
    val = df_sorted.iloc[train_end:val_end].copy()
    test = df_sorted.iloc[val_end:].copy()

    # Safety: every split must contain fraud cases
    for name, split in [("train", train), ("val", val), ("test", test)]:
        fraud_count = int(split[TARGET].sum())
        assert fraud_count > 0, f"{name} split has zero fraud cases!"

    return train, val, test


def print_split_info(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> None:
    """Pretty-print sizes and fraud rates for each split."""
    total_all = len(train) + len(val) + len(test)
    for name, split in [("Train", train), ("Validation", val), ("Test", test)]:
        total = len(split)
        fraud = int(split[TARGET].sum())
        rate = fraud / total * 100
        pct = total / total_all * 100
        print(
            f"  {name:<12} {total:>7,} rows ({pct:>5.1f}%) | "
            f"{fraud:>4} frauds | fraud rate {rate:.4f}%"
        )


# ── CLI entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FraudShield - Data Loading & Time-Based Split")
    print("=" * 60)

    print("\nLoading dataset ...")
    df = load_data()
    print(
        f"Total: {len(df):,} rows, {int(df[TARGET].sum())} frauds "
        f"({df[TARGET].mean() * 100:.4f}% fraud rate)\n"
    )

    print("Performing time-based split (70 / 15 / 15) ...")
    train, val, test = time_based_split(df)
    print_split_info(train, val, test)

    print("\n[OK] All splits contain fraud cases.  Data is ready.")
    print("=" * 60)
