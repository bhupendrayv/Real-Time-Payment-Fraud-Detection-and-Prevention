"""Download the ULB Credit Card Fraud Detection dataset from Kaggle.

Usage:
    python scripts/download_data.py
"""
from __future__ import annotations

import shutil
from pathlib import Path


def download_dataset(target_dir: Path | None = None) -> Path | None:
    """Download creditcard.csv from Kaggle into data/.

    Uses ``kagglehub`` for automatic download.  Falls back to
    manual instructions if authentication is missing.

    Returns:
        Path to the downloaded CSV, or None on failure.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "creditcard.csv"

    if target_path.exists():
        print(f"[OK] Dataset already exists at {target_path}")
        return target_path

    print("Downloading Credit Card Fraud Detection dataset from Kaggle ...")
    try:
        import kagglehub

        cached_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        # kagglehub returns a directory; find the CSV inside it
        source: Path | None = None
        for candidate in Path(cached_path).rglob("creditcard.csv"):
            source = candidate
            break

        if source is None:
            print(f"[ERROR] Could not find creditcard.csv in {cached_path}")
            _print_manual_instructions(target_dir)
            return None

        shutil.copy2(source, target_path)
        print(f"[OK] Saved to {target_path}")
        return target_path

    except Exception as exc:
        print(f"[ERROR] Download failed: {exc}")
        _print_manual_instructions(target_dir)
        return None


def _print_manual_instructions(target_dir: Path) -> None:
    """Print manual download steps."""
    print("\nManual download:")
    print("  1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  2. Click 'Download' → extract creditcard.csv")
    print(f"  3. Place it in {target_dir / 'creditcard.csv'}")


if __name__ == "__main__":
    import pandas as pd

    path = download_dataset()
    if path and path.exists():
        df = pd.read_csv(path)
        print("\nDataset summary:")
        print(f"  Rows:    {len(df):,}")
        print(f"  Columns: {df.shape[1]}")
        print(
            f"  Frauds:  {int(df['Class'].sum()):,} "
            f"({df['Class'].mean() * 100:.3f}%)"
        )
