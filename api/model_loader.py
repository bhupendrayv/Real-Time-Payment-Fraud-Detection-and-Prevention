"""Model and SHAP explainer lifecycle manager.

Loads production model artifacts, decision threshold, scaler,
and shap.TreeExplainer exactly once at startup.
"""
from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from api.config import MODEL_VERSION
from ml.config import ARTIFACTS_DIR, PCA_FEATURES
from ml.features import FeaturePreprocessor
from ml.train_autoencoder import FraudAutoencoder


class ModelManager:
    """Manages loaded ML artifacts and provides fast in-memory inference."""

    def __init__(self) -> None:
        self.model: Any = None
        self.threshold: float = 0.5
        self.preprocessor: FeaturePreprocessor | None = None
        self.autoencoder: FraudAutoencoder | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.feature_names: list[str] = []
        self.model_version: str = MODEL_VERSION
        self.is_hybrid: bool = False
        self.is_loaded: bool = False

    def load(self) -> None:
        """Load artifacts from ml/artifacts/ once at server startup."""
        if self.is_loaded:
            return

        preprocessor_path = ARTIFACTS_DIR / "preprocessor.joblib"
        best_model_path = ARTIFACTS_DIR / "best_model.joblib"
        hybrid_model_path = ARTIFACTS_DIR / "hybrid_model.joblib"
        threshold_path = ARTIFACTS_DIR / "threshold.json"
        autoencoder_path = ARTIFACTS_DIR / "autoencoder.pt"

        # 1. Load fitted preprocessor
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {preprocessor_path}")
        self.preprocessor = FeaturePreprocessor.load(preprocessor_path)

        # 2. Load model (prefer best_model.joblib, fallback to hybrid_model.joblib)
        target_model_path = best_model_path if best_model_path.exists() else hybrid_model_path
        if not target_model_path.exists():
            raise FileNotFoundError(f"No trained model artifact found at {target_model_path}")

        self.model = joblib.load(target_model_path)
        self.is_hybrid = getattr(self.model, "n_features_in_", 30) == 31

        # 3. Load threshold
        if threshold_path.exists():
            with open(threshold_path, "r", encoding="utf-8") as f:
                thresh_data = json.load(f)
            self.threshold = float(thresh_data.get("threshold", 0.5))
        else:
            self.threshold = 0.5

        # 4. If hybrid model, load PyTorch autoencoder weights
        if self.is_hybrid:
            if not autoencoder_path.exists():
                raise FileNotFoundError(f"Hybrid model requires autoencoder weights at {autoencoder_path}")
            self.autoencoder = FraudAutoencoder(input_dim=30)
            self.autoencoder.load_state_dict(
                torch.load(autoencoder_path, map_location=torch.device("cpu"), weights_only=True)
            )
            self.autoencoder.eval()

        # 5. Define feature ordering
        self.feature_names = ["Time", "Amount"] + PCA_FEATURES
        if self.is_hybrid:
            self.feature_names.append("recon_error")

        # 6. Initialize SHAP TreeExplainer once
        self.explainer = shap.TreeExplainer(self.model)
        self.is_loaded = True

    def _compute_reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error with frozen PyTorch Autoencoder."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder not loaded for hybrid model")
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_scaled.astype(np.float32))
            reconstructed = self.autoencoder(tensor_x)
            mse = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()
        return mse.astype(np.float32)

    def prepare_input(self, tx_dict: dict[str, Any]) -> np.ndarray:
        """Preprocess single transaction dict into scaled numpy array matching training format."""
        if self.preprocessor is None:
            raise RuntimeError("ModelManager has not been initialized. Call load() first.")

        raw_cols = ["Time", "Amount"] + PCA_FEATURES
        df = pd.DataFrame([tx_dict])[raw_cols]
        X_scaled = self.preprocessor.transform(df)

        if self.is_hybrid:
            recon_err = self._compute_reconstruction_error(X_scaled)
            X_model = np.column_stack([X_scaled, recon_err])
        else:
            X_model = X_scaled

        return np.ascontiguousarray(X_model, dtype=np.float32)

    def predict(
        self, tx_dict: dict[str, Any], explain: bool = True, top_k: int = 5
    ) -> tuple[float, bool, list[dict[str, Any]]]:
        """Perform fraud probability scoring and optional SHAP explanation.

        Returns:
            (fraud_score, is_fraud, explanations)
        """
        X_input = self.prepare_input(tx_dict)

        # 1. Scoring (probability of Class 1)
        proba = float(self.model.predict_proba(X_input)[0, 1])
        is_fraud = bool(proba >= self.threshold)

        # 2. SHAP explanation
        explanations: list[dict[str, Any]] = []
        if explain and self.explainer is not None:
            shap_values = self.explainer.shap_values(X_input)
            if isinstance(shap_values, list):
                raw_contribs = shap_values[1][0]
            elif shap_values.ndim == 3:
                raw_contribs = shap_values[0, :, 1]
            elif shap_values.ndim == 2:
                raw_contribs = shap_values[0]
            else:
                raw_contribs = np.array(shap_values).flatten()

            pairs = [
                {"feature": name, "contribution": float(val)}
                for name, val in zip(self.feature_names, raw_contribs)
            ]
            pairs.sort(key=lambda item: abs(item["contribution"]), reverse=True)
            explanations = pairs[:top_k]

        return proba, is_fraud, explanations


# Singleton manager instance
model_manager = ModelManager()
