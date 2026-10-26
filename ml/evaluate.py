"""Evaluation metrics, cost-based threshold selection, and reporting.

Strictly adheres to quality rules:
- PR-AUC is primary metric (never accuracy)
- Cost function: cost = FN * avg_fraud_amount + FP * review_cost
- Threshold is selected on validation set only, then evaluated on test set once.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ml.config import ARTIFACTS_DIR


DEFAULT_REVIEW_COST: float = 10.0  # Assumed cost to manually review a flagged transaction


def calculate_expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray | None = None,
    avg_fraud_amount: float = 122.0,
    review_cost: float = DEFAULT_REVIEW_COST,
) -> float:
    """Calculate business cost of model decisions.

    cost = FN * avg_fraud_amount + FP * review_cost
    If individual amounts are provided for FN cases, exact lost amounts can be summed.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if amounts is not None:
        fn_mask = (y_true == 1) & (y_pred == 0)
        fn_loss = float(amounts[fn_mask].sum())
    else:
        fn_loss = float(fn * avg_fraud_amount)

    fp_cost = float(fp * review_cost)
    return fn_loss + fp_cost


def recall_at_fixed_fpr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    target_fpr: float = 0.001,
) -> float:
    """Calculate recall achieved at a fixed False Positive Rate (e.g., 0.1% FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    # Find the largest FPR that is <= target_fpr
    valid_idx = np.where(fpr <= target_fpr)[0]
    if len(valid_idx) == 0:
        return 0.0
    best_idx = valid_idx[-1]
    return float(tpr[best_idx])


def find_optimal_threshold(
    y_val: np.ndarray,
    y_val_proba: np.ndarray,
    amounts_val: np.ndarray | None = None,
    avg_fraud_amount: float = 122.0,
    review_cost: float = DEFAULT_REVIEW_COST,
    num_thresholds: int = 200,
) -> tuple[float, float]:
    """Find the probability threshold that minimizes expected cost on the validation set.

    Returns:
        (optimal_threshold, min_cost)
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    best_thresh = 0.5
    min_cost = float("inf")

    for t in thresholds:
        y_pred = (y_val_proba >= t).astype(int)
        cost = calculate_expected_cost(
            y_val, y_pred, amounts=amounts_val, avg_fraud_amount=avg_fraud_amount, review_cost=review_cost
        )
        if cost < min_cost:
            min_cost = cost
            best_thresh = float(t)

    return best_thresh, min_cost


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    amounts: np.ndarray | None = None,
    avg_fraud_amount: float = 122.0,
    review_cost: float = DEFAULT_REVIEW_COST,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics for a model's predictions."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pr_auc = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    rec_at_fpr = recall_at_fixed_fpr(y_true, y_proba, target_fpr=0.001)

    expected_cost = calculate_expected_cost(
        y_true, y_pred, amounts=amounts, avg_fraud_amount=avg_fraud_amount, review_cost=review_cost
    )

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_at_0.1pct_fpr": rec_at_fpr,
        "expected_cost": expected_cost,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_threshold_config(
    threshold: float,
    metadata: dict[str, Any],
    path: Path | str | None = None,
) -> Path:
    """Save the chosen threshold and metadata to artifacts."""
    target_path = Path(path) if path else ARTIFACTS_DIR / "threshold.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": float(threshold),
        "metadata": metadata,
    }
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return target_path
