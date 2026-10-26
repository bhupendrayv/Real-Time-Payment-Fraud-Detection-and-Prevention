"""Feature preprocessing for FraudShield.

Applies RobustScaler to 'Time' and 'Amount' (handling extreme outliers)
while passing through V1-V28 (already PCA-transformed).

CRITICAL: The preprocessor is fit ONLY on the training split to avoid data leakage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler

from ml.config import ARTIFACTS_DIR, PCA_FEATURES


PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Preprocessor for payment transactions.

    Fits RobustScaler on 'Time' and 'Amount' on training data only.
    V1-V28 are passed through untouched.
    """

    def __init__(self) -> None:
        self.scaler = RobustScaler()
        self.feature_names_out: list[str] = []
        self.is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: Any = None) -> "FeaturePreprocessor":
        """Fit the scaler on training features only."""
        cols_to_scale = ["Time", "Amount"]
        for col in cols_to_scale:
            if col not in X.columns:
                raise ValueError(f"Required column '{col}' missing from input DataFrame")

        self.scaler.fit(X[cols_to_scale])
        self.feature_names_out = cols_to_scale + [c for c in PCA_FEATURES if c in X.columns]
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features into a scaled numpy array."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform.")

        cols_to_scale = ["Time", "Amount"]
        scaled_part = self.scaler.transform(X[cols_to_scale])
        pca_cols = [c for c in PCA_FEATURES if c in X.columns]
        pca_part = X[pca_cols].to_numpy()

        out = np.hstack([scaled_part, pca_part])
        return np.ascontiguousarray(out, dtype=np.float32)

    def fit_transform(self, X: pd.DataFrame, y: Any = None) -> np.ndarray:
        """Fit on X and transform X."""
        return self.fit(X, y).transform(X)

    def save(self, path: Path | str | None = None) -> Path:
        """Serialize fitted preprocessor to disk."""
        target_path = Path(path) if path else PREPROCESSOR_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target_path)
        return target_path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "FeaturePreprocessor":
        """Load serialized preprocessor from disk."""
        target_path = Path(path) if path else PREPROCESSOR_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"No preprocessor artifact found at {target_path}")
        instance = joblib.load(target_path)
        return instance


def prepare_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FeaturePreprocessor]:
    """Fit preprocessor on train, then transform train, val, test.

    Returns:
        (X_train, X_val, X_test, fitted_preprocessor)
    """
    feature_cols = ["Time", "Amount"] + PCA_FEATURES
    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(train[feature_cols])
    X_val = preprocessor.transform(val[feature_cols])
    X_test = preprocessor.transform(test[feature_cols])

    return X_train, X_val, X_test, preprocessor
