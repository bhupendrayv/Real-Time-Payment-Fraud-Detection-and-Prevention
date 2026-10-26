"""Statistical data and model drift detection engine for FraudShield.

Implements:
- Population Stability Index (PSI) per feature and on model scores.
- Kolmogorov-Smirnov (KS) two-sample test against reference training samples.
- Rolling window evaluation from recent transaction audit logs.
- Multi-tier alert classification: "ok", "warning", "drift", "insufficient_data".
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sqlalchemy.orm import Session

from api.db import TransactionRecord
from ml.config import (
    DRIFT_FEATURES,
    DRIFT_MIN_ROWS,
    DRIFT_WINDOW_SIZE,
    PSI_OK_THRESHOLD,
    PSI_WARNING_THRESHOLD,
    REFERENCE_STATS_PATH,
)


def calculate_psi(
    actual: np.ndarray,
    expected_edges: list[float],
    expected_proportions: list[float],
    epsilon: float = 1e-6,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and actual data.

    Formula:
        PSI = sum((actual_prop - expected_prop) * ln((actual_prop + eps) / (expected_prop + eps)))

    Args:
        actual: 1D numpy array of current live feature values.
        expected_edges: Bin edges used during training baseline.
        expected_proportions: Fraction of reference data in each bin.
        epsilon: Small constant to prevent log(0) and division by zero.

    Returns:
        float: Computed PSI value (non-negative).
    """
    if len(actual) == 0:
        return 0.0

    # Ensure finite edges for np.histogram while capturing extreme tails in first & last bin
    edges = np.array(expected_edges, dtype=float)
    if np.isneginf(edges[0]):
        edges[0] = -1e12
    if np.isposinf(edges[-1]):
        edges[-1] = 1e12

    # Clip actual data to edges so extremes fall cleanly into outer bins
    clipped_actual = np.clip(actual, edges[0], edges[-1])
    counts, _ = np.histogram(clipped_actual, bins=edges)
    actual_proportions = counts / float(len(actual))

    expected = np.array(expected_proportions, dtype=float)
    actual_prop = np.array(actual_proportions, dtype=float)

    # Protect against zero proportions
    expected_adj = expected + epsilon
    actual_adj = actual_prop + epsilon

    psi_bins = (actual_adj - expected_adj) * np.log(actual_adj / expected_adj)
    return float(np.sum(psi_bins))


def calculate_ks(actual: np.ndarray, reference_sample: list[float]) -> tuple[float, float]:
    """Perform two-sample Kolmogorov-Smirnov test.

    Returns:
        (ks_statistic, p_value)
    """
    if len(actual) == 0 or len(reference_sample) == 0:
        return 0.0, 1.0

    res = stats.ks_2samp(actual, reference_sample)
    return float(res.statistic), float(res.pvalue)


def map_psi_to_status(psi: float) -> str:
    """Map PSI value to operational alert status.

    - PSI < 0.10: "ok" (distribution is stable)
    - 0.10 <= PSI <= 0.25: "warning" (slight shift, observe closely)
    - PSI > 0.25: "drift" (significant distribution shift, action required)
    """
    if psi < PSI_OK_THRESHOLD:
        return "ok"
    elif psi <= PSI_WARNING_THRESHOLD:
        return "warning"
    else:
        return "drift"


class DriftMonitor:
    """Evaluates data and prediction score drift against training baseline."""

    def __init__(self, reference_path: Path = REFERENCE_STATS_PATH) -> None:
        self.reference_path = reference_path
        self._reference_data: dict[str, Any] | None = None
        self.load_reference()

    def load_reference(self) -> None:
        """Load reference statistics from JSON."""
        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference stats not found at {self.reference_path}. "
                "Run 'python -m ml.build_reference' first."
            )
        with open(self.reference_path, "r", encoding="utf-8") as f:
            self._reference_data = json.load(f)

    @property
    def reference(self) -> dict[str, Any]:
        if self._reference_data is None:
            self.load_reference()
        assert self._reference_data is not None
        return self._reference_data

    @property
    def reference_flagged_rate(self) -> float:
        return float(self.reference.get("score_reference", {}).get("reference_flagged_rate", 0.0))

    def evaluate_features_and_scores(
        self,
        features_dict: dict[str, np.ndarray],
        fraud_scores: np.ndarray,
        is_fraud_array: np.ndarray | None = None,
        window_size: int = DRIFT_WINDOW_SIZE,
        min_rows: int = DRIFT_MIN_ROWS,
    ) -> dict[str, Any]:
        """Compute drift metrics directly from numpy arrays.

        Useful for unit tests and direct evaluations.
        """
        n_samples = len(fraud_scores)
        computed_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

        if n_samples < min_rows:
            return {
                "status": "insufficient_data",
                "window_size": n_samples,
                "min_rows_required": min_rows,
                "computed_at": computed_at,
                "score_psi": 0.0,
                "flagged_rate": 0.0,
                "reference_flagged_rate": round(self.reference_flagged_rate, 5),
                "top_drifting_features": [],
            }

        ref_features = self.reference.get("features", {})
        feature_results: list[dict[str, Any]] = []

        for feat_name in DRIFT_FEATURES:
            if feat_name not in ref_features or feat_name not in features_dict:
                continue

            ref_f = ref_features[feat_name]
            feat_values = features_dict[feat_name]

            psi_val = calculate_psi(
                actual=feat_values,
                expected_edges=ref_f["bin_edges"],
                expected_proportions=ref_f["proportions"],
            )

            ks_stat, ks_pval = calculate_ks(
                actual=feat_values,
                reference_sample=ref_f.get("ks_sample", []),
            )

            status = map_psi_to_status(psi_val)
            feature_results.append({
                "feature": feat_name,
                "psi": round(psi_val, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 5),
                "status": status,
            })

        # Sort features by PSI descending
        feature_results.sort(key=lambda x: x["psi"], reverse=True)
        top_5_features = feature_results[:5]

        # Score drift
        ref_score = self.reference.get("score_reference", {})
        score_edges = ref_score.get("bin_edges", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        score_props = ref_score.get("proportions", [0.1] * 10)
        score_psi = calculate_psi(fraud_scores, score_edges, score_props)

        # Flagged rate
        if is_fraud_array is not None and len(is_fraud_array) > 0:
            live_flagged_rate = float(np.mean(is_fraud_array))
        else:
            threshold = float(self.reference.get("metadata", {}).get("threshold", 0.5))
            live_flagged_rate = float(np.mean(fraud_scores >= threshold))

        # Overall status is worst feature status (or score drift status)
        statuses = [f["status"] for f in feature_results]
        if "drift" in statuses or score_psi > PSI_WARNING_THRESHOLD:
            overall_status = "drift"
        elif "warning" in statuses or score_psi >= PSI_OK_THRESHOLD:
            overall_status = "warning"
        else:
            overall_status = "ok"

        return {
            "status": overall_status,
            "window_size": n_samples,
            "min_rows_required": min_rows,
            "computed_at": computed_at,
            "score_psi": round(score_psi, 4),
            "flagged_rate": round(live_flagged_rate, 5),
            "reference_flagged_rate": round(self.reference_flagged_rate, 5),
            "top_drifting_features": top_5_features,
        }

    def evaluate_from_db(
        self,
        db: Session,
        window_size: int = DRIFT_WINDOW_SIZE,
        min_rows: int = DRIFT_MIN_ROWS,
    ) -> dict[str, Any]:
        """Fetch the most recent N transactions from DB and compute drift."""
        records = (
            db.query(TransactionRecord)
            .order_by(TransactionRecord.created_at.desc())
            .limit(window_size)
            .all()
        )

        n = len(records)
        if n < min_rows:
            return {
                "status": "insufficient_data",
                "window_size": n,
                "min_rows_required": min_rows,
                "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "score_psi": 0.0,
                "flagged_rate": 0.0,
                "reference_flagged_rate": round(self.reference_flagged_rate, 5),
                "top_drifting_features": [],
            }

        # Extract features and scores
        scores = np.array([r.fraud_score for r in records], dtype=float)
        is_fraud = np.array([r.is_fraud for r in records], dtype=bool)

        features_dict: dict[str, np.ndarray] = {feat: [] for feat in DRIFT_FEATURES}
        for r in records:
            f_json = r.features or {}
            for feat in DRIFT_FEATURES:
                val = f_json.get(feat, 0.0)
                features_dict[feat].append(val)

        features_np = {k: np.array(v, dtype=float) for k, v in features_dict.items()}

        return self.evaluate_features_and_scores(
            features_dict=features_np,
            fraud_scores=scores,
            is_fraud_array=is_fraud,
            window_size=window_size,
            min_rows=min_rows,
        )


# Global drift monitor instance
drift_monitor = DriftMonitor()
