"""Unit tests for Phase 4: Model Explainability with SHAP."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from ml.data import load_data, time_based_split
from ml.explain import explain, get_explainer


@pytest.fixture(scope="session")
def test_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load test dataset and separate into fraud and legitimate samples."""
    df = load_data()
    _, _, test_df = time_based_split(df)
    fraud_df = test_df[test_df["Class"] == 1]
    non_fraud_df = test_df[test_df["Class"] == 0]
    return fraud_df, non_fraud_df


def test_explain_output_has_exactly_five_items(test_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Test that explain() returns a list with exactly 5 items."""
    _, non_fraud_df = test_data
    sample_tx = non_fraud_df.iloc[0].to_dict()

    result = explain(sample_tx)

    assert isinstance(result, list), "Output must be a list"
    assert len(result) == 5, f"Expected exactly 5 items, got {len(result)}"


def test_explain_item_structure(test_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Test that each item in explain() has 'feature' and 'contribution' keys."""
    _, non_fraud_df = test_data
    sample_tx = non_fraud_df.iloc[1].to_dict()

    result = explain(sample_tx)

    for item in result:
        assert isinstance(item, dict), "Each item must be a dictionary"
        assert "feature" in item, "Item must contain 'feature' key"
        assert "contribution" in item, "Item must contain 'contribution' key"
        assert isinstance(item["feature"], str), "Feature name must be a string"
        assert len(item["feature"]) > 0, "Feature name must not be empty"


def test_explain_contributions_are_finite_numbers(test_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Test that each contribution is a valid finite float (no NaN or Inf)."""
    _, non_fraud_df = test_data
    sample_tx = non_fraud_df.iloc[2].to_dict()

    result = explain(sample_tx)

    for item in result:
        val = item["contribution"]
        assert isinstance(val, (float, int, np.floating)), f"Contribution {val} is not a number"
        assert not math.isnan(val), f"Contribution {val} is NaN"
        assert not math.isinf(val), f"Contribution {val} is infinite"


def test_explain_sort_order_by_absolute_contribution(test_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Test that output items are ordered by absolute contribution descending."""
    _, non_fraud_df = test_data
    sample_tx = non_fraud_df.iloc[3].to_dict()

    result = explain(sample_tx)
    abs_values = [abs(item["contribution"]) for item in result]

    # Verify descending order
    for i in range(len(abs_values) - 1):
        assert abs_values[i] >= abs_values[i + 1] - 1e-9, "Items must be sorted by |contribution| descending"


def test_known_fraud_row_has_positive_total_contribution(
    test_data: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    """Test that a known fraud row from the test set has a positive total SHAP contribution.

    Positive SHAP contribution pushes model log-odds toward fraud (Class 1).
    """
    fraud_df, _ = test_data
    explainer = get_explainer()

    # Test across multiple known fraud transactions
    tested_frauds = 0
    positive_sums = 0

    for idx in range(min(10, len(fraud_df))):
        fraud_tx = fraud_df.iloc[idx].to_dict()
        X_arr = explainer.prepare_input(fraud_tx)
        shap_vals = explainer.explainer.shap_values(X_arr)

        if isinstance(shap_vals, list):
            vals = shap_vals[1][0]
        elif shap_vals.ndim == 3:
            vals = shap_vals[0, :, 1]
        elif shap_vals.ndim == 2:
            vals = shap_vals[0]
        else:
            vals = np.array(shap_vals).flatten()

        total_sum = float(np.sum(vals))
        tested_frauds += 1
        if total_sum > 0:
            positive_sums += 1

    # In our test set, known frauds push heavily toward fraud prediction
    assert tested_frauds > 0
    assert positive_sums / tested_frauds >= 0.8, (
        f"Expected at least 80% of confirmed fraud rows to have positive total contribution, "
        f"got {positive_sums}/{tested_frauds}"
    )


def test_explain_handles_dataframe_and_series(test_data: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Test that explain accepts pandas Series, DataFrame, or dict seamlessly."""
    fraud_df, _ = test_data
    tx_series = fraud_df.iloc[0]
    tx_df = fraud_df.iloc[[0]]
    tx_dict = tx_series.to_dict()

    res_series = explain(tx_series)
    res_df = explain(tx_df)
    res_dict = explain(tx_dict)

    assert len(res_series) == 5
    assert len(res_df) == 5
    assert len(res_dict) == 5

    # Features and contributions should match exactly across input types
    for r_s, r_d, r_dict in zip(res_series, res_df, res_dict):
        assert r_s["feature"] == r_d["feature"] == r_dict["feature"]
        assert pytest.approx(r_s["contribution"], abs=1e-5) == r_d["contribution"]
        assert pytest.approx(r_s["contribution"], abs=1e-5) == r_dict["contribution"]
