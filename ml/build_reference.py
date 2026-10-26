"""Build reference statistics from the TRAINING split and validation set for drift monitoring.

Computes:
1. Per-feature baseline distributions (bin edges, proportions, mean, std, and KS sample) from the TRAINING split only.
2. Validation fraud score distribution and reference flagged rate.

Saves output to ml/artifacts/reference_stats.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from api.model_loader import ModelManager
from ml.config import (
    DRIFT_FEATURES,
    RAW_DATA_PATH,
    REFERENCE_STATS_PATH,
    SEED,
)
from ml.data import load_data, time_based_split


def compute_feature_reference(
    series: pd.Series,
    n_bins: int = 10,
    ks_sample_size: int = 2000,
    seed: int = SEED,
) -> dict[str, Any]:
    """Compute bin edges, proportions, mean, std, and sample for a single feature."""
    values = series.dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        raise ValueError(f"Feature series {series.name} has no valid numeric values.")

    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    if std_val == 0:
        std_val = 1e-6

    # Decile bin edges based on training split
    quantiles = np.linspace(0, 100, n_bins + 1)
    raw_edges = np.percentile(values, quantiles)
    # Ensure strictly increasing edges by adding tiny jitter if duplicated
    edges = list(raw_edges)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-5

    # Replace outer boundaries with -inf and inf for robust binning
    bin_edges = [-float("inf")] + [float(e) for e in edges[1:-1]] + [float("inf")]

    # Compute reference counts and proportions
    counts, _ = np.histogram(values, bins=edges)
    proportions = [float(c / n) for c in counts]

    # Draw deterministic sample for KS testing
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(n, size=min(ks_sample_size, n), replace=False)
    ks_sample = [float(x) for x in values[sample_indices]]

    return {
        "mean": mean_val,
        "std": std_val,
        "bin_edges": bin_edges,
        "proportions": proportions,
        "ks_sample": ks_sample,
    }


def build_reference_stats(
    raw_data_path: Path = RAW_DATA_PATH,
    output_path: Path = REFERENCE_STATS_PATH,
) -> dict[str, Any]:
    """Extract reference stats from training split and validation predictions."""
    print("Loading data splits...")
    df = load_data(str(raw_data_path))
    train_df, val_df, _ = time_based_split(df)

    print(f"Training split rows: {len(train_df):,} | Validation split rows: {len(val_df):,}")

    # 1. Feature reference distributions from TRAINING split only
    features_stats: dict[str, Any] = {}
    for col in DRIFT_FEATURES:
        if col not in train_df.columns:
            raise KeyError(f"Expected feature '{col}' missing from training data.")
        features_stats[col] = compute_feature_reference(train_df[col])

    # 2. Validation score distribution & reference flagged rate using ModelManager
    print("Scoring validation split with production model...")
    mgr = ModelManager()
    mgr.load()

    raw_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    val_features = val_df[raw_cols]
    assert mgr.preprocessor is not None, "Preprocessor not loaded"
    X_scaled = mgr.preprocessor.transform(val_features)
    if mgr.is_hybrid:
        recon_err = mgr._compute_reconstruction_error(X_scaled)
        X_model = np.column_stack([X_scaled, recon_err])
    else:
        X_model = X_scaled

    val_scores_arr = mgr.model.predict_proba(X_model)[:, 1].astype(float)

    # Score reference bins: 10 uniform buckets from 0.0 to 1.0
    score_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    score_counts, _ = np.histogram(val_scores_arr, bins=score_edges)
    score_proportions = [float(c / len(val_scores_arr)) for c in score_counts]

    flagged_mask = val_scores_arr >= mgr.threshold
    ref_flagged_rate = float(np.mean(flagged_mask))

    ref_payload: dict[str, Any] = {
        "metadata": {
            "created_from": "train_split_only",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "threshold": mgr.threshold,
            "model_version": mgr.model_version,
        },
        "score_reference": {
            "bin_edges": score_edges,
            "proportions": score_proportions,
            "reference_flagged_rate": ref_flagged_rate,
        },
        "features": features_stats,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ref_payload, f, indent=2)

    print(f"[OK] Reference stats successfully saved to {output_path}")
    print(f"     Threshold: {mgr.threshold:.4f} | Validation flagged rate: {ref_flagged_rate:.4%}")
    return ref_payload


if __name__ == "__main__":
    build_reference_stats()
