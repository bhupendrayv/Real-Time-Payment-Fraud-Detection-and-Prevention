"""Extract a compact demo sample (2,000 transactions) from the test split.

Ensures all test-split fraud records are preserved for rich demonstration and testing,
while keeping file size small (~280 KB) so it can be committed and used without Kaggle raw data.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from ml.config import RAW_DATA_PATH, TEST_RATIO, TARGET

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"
OUTPUT_FILE = OUTPUT_DIR / "demo_transactions.csv"


def create_demo_sample(sample_size: int = 2000, random_state: int = 42) -> Path:
    """Create a sample from the test split containing all test fraud records."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    df = df.sort_values(by="Time").reset_index(drop=True)

    # Replicate exact time-based split
    test_size = int(len(df) * TEST_RATIO)
    test_df = df.iloc[-test_size:].copy().reset_index(drop=True)

    # Separate fraud and non-fraud in test split
    frauds = test_df[test_df[TARGET] == 1]
    non_frauds = test_df[test_df[TARGET] == 0]

    num_frauds = len(frauds)
    num_non_frauds_needed = max(0, sample_size - num_frauds)

    sampled_non_frauds = non_frauds.sample(
        n=min(num_non_frauds_needed, len(non_frauds)),
        random_state=random_state,
    )

    demo_df = pd.concat([frauds, sampled_non_frauds], axis=0)
    # Sort by time so stream ordering remains natural
    demo_df = demo_df.sort_values(by="Time").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    demo_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Created demo sample at {OUTPUT_FILE}")
    print(f"Total rows: {len(demo_df)}, Frauds: {demo_df[TARGET].sum()}, Non-frauds: {(demo_df[TARGET] == 0).sum()}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    return OUTPUT_FILE


if __name__ == "__main__":
    create_demo_sample()
