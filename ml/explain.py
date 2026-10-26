"""Explainability module for FraudShield payment fraud detection.

Provides:
- FraudExplainer: Singleton/reusable explainer loading model and shap.TreeExplainer once.
- explain(transaction): Computes top 5 feature contributions (positive = pushes toward fraud,
  negative = pushes toward legitimate).
- generate_global_explanations(): Saves global beeswarm and feature importance bar plots.
- benchmark_explain_latency(): Measures mean and p95 explain latency over 100 transactions.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from ml.config import ARTIFACTS_DIR, DOCS_DIR, PCA_FEATURES, SEED
from ml.data import load_data, time_based_split
from ml.features import FeaturePreprocessor
from ml.train_autoencoder import FraudAutoencoder


class FraudExplainer:
    """Reusable TreeExplainer wrapper for transaction fraud explainability.

    Caches the preprocessor, autoencoder (if hybrid), XGBoost model,
    and TreeExplainer instance to avoid redundant initialization overhead.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        preprocessor_path: Path | str | None = None,
        autoencoder_path: Path | str | None = None,
    ) -> None:
        self.preprocessor_path = (
            Path(preprocessor_path) if preprocessor_path else ARTIFACTS_DIR / "preprocessor.joblib"
        )
        self.autoencoder_path = (
            Path(autoencoder_path) if autoencoder_path else ARTIFACTS_DIR / "autoencoder.pt"
        )

        # 1. Load fitted preprocessor
        self.preprocessor = FeaturePreprocessor.load(self.preprocessor_path)

        # 2. Determine and load production model
        if model_path is not None:
            self.model_path = Path(model_path)
        else:
            # Check threshold.json or available artifacts
            default_best = ARTIFACTS_DIR / "best_model.joblib"
            hybrid_best = ARTIFACTS_DIR / "hybrid_model.joblib"
            self.model_path = default_best if default_best.exists() else hybrid_best

        self.model = joblib.load(self.model_path)
        self.is_hybrid: bool = getattr(self.model, "n_features_in_", 30) == 31

        # 3. If hybrid model, load PyTorch Autoencoder to compute reconstruction error
        self.autoencoder: FraudAutoencoder | None = None
        if self.is_hybrid:
            if not self.autoencoder_path.exists():
                raise FileNotFoundError(
                    f"Hybrid model requires autoencoder weights at {self.autoencoder_path}"
                )
            self.autoencoder = FraudAutoencoder(input_dim=30)
            self.autoencoder.load_state_dict(
                torch.load(self.autoencoder_path, map_location=torch.device("cpu"), weights_only=True)
            )
            self.autoencoder.eval()

        # 4. Feature names aligned with model input
        # FeaturePreprocessor produces: ['Time', 'Amount', 'V1', ..., 'V28']
        self.feature_names: list[str] = ["Time", "Amount"] + PCA_FEATURES
        if self.is_hybrid:
            self.feature_names.append("recon_error")

        # 5. Build TreeExplainer once
        # Using feature_perturbation="tree_path_dependent" ensures fast exact TreeSHAP computation
        self.explainer = shap.TreeExplainer(self.model)

    def _compute_reconstruction_error(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute MSE reconstruction error using the frozen Autoencoder."""
        if self.autoencoder is None:
            raise RuntimeError("Autoencoder is not initialized for non-hybrid model.")
        with torch.no_grad():
            tensor_x = torch.from_numpy(X_scaled.astype(np.float32))
            reconstructed = self.autoencoder(tensor_x)
            mse = torch.mean((tensor_x - reconstructed) ** 2, dim=1).numpy()
        return mse.astype(np.float32)

    def prepare_input(
        self, transaction: Mapping[str, Any] | pd.DataFrame | pd.Series | np.ndarray
    ) -> np.ndarray:
        """Convert a single transaction dict/Series/DataFrame into model-ready ndarray."""
        if isinstance(transaction, Mapping):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.Series):
            df = pd.DataFrame([transaction.to_dict()])
        elif isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        elif isinstance(transaction, np.ndarray):
            if transaction.ndim == 1:
                return transaction.reshape(1, -1)
            return transaction
        else:
            raise TypeError(
                f"Unsupported transaction type: {type(transaction)}. "
                "Expected dict, pd.Series, pd.DataFrame, or np.ndarray."
            )

        # Verify raw columns exist
        raw_cols = ["Time", "Amount"] + PCA_FEATURES
        missing = [col for col in raw_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Transaction missing required columns: {missing}")

        # Preprocess features (Time & Amount scaled with RobustScaler, V1-V28 passthrough)
        X_scaled = self.preprocessor.transform(df[raw_cols])

        if self.is_hybrid:
            recon_err = self._compute_reconstruction_error(X_scaled)
            X_model = np.column_stack([X_scaled, recon_err])
        else:
            X_model = X_scaled

        return np.ascontiguousarray(X_model, dtype=np.float32)

    def explain(
        self,
        transaction: Mapping[str, Any] | pd.DataFrame | pd.Series | np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Explain a transaction's fraud score.

        Returns:
            List of { "feature": str, "contribution": float } ordered by absolute contribution descending.
            Positive contribution pushes toward fraud (Class 1),
            negative contribution pushes toward legitimate (Class 0).
        """
        X_arr = self.prepare_input(transaction)

        # TreeExplainer on XGBoost returns array of shape (N, num_features)
        shap_values = self.explainer.shap_values(X_arr)

        if isinstance(shap_values, list):
            # Binary classification list format [class_0, class_1]
            raw_contribs = shap_values[1][0]
        elif shap_values.ndim == 3:
            raw_contribs = shap_values[0, :, 1]
        elif shap_values.ndim == 2:
            raw_contribs = shap_values[0]
        else:
            raw_contribs = np.array(shap_values).flatten()

        contrib_pairs = [
            {"feature": name, "contribution": float(val)}
            for name, val in zip(self.feature_names, raw_contribs)
        ]

        # Sort by absolute contribution descending
        contrib_pairs.sort(key=lambda item: abs(item["contribution"]), reverse=True)
        return contrib_pairs[:top_k]


# Global singleton instance for module-level explain() calls
_EXPLAINER_INSTANCE: FraudExplainer | None = None


def get_explainer() -> FraudExplainer:
    """Return the cached FraudExplainer singleton."""
    global _EXPLAINER_INSTANCE
    if _EXPLAINER_INSTANCE is None:
        _EXPLAINER_INSTANCE = FraudExplainer()
    return _EXPLAINER_INSTANCE


def explain(
    transaction: Mapping[str, Any] | pd.DataFrame | pd.Series | np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Convenience module function for explaining a single transaction.

    Returns the top_k (default 5) features with highest absolute SHAP impact.
    """
    explainer = get_explainer()
    return explainer.explain(transaction, top_k=top_k)


def generate_global_explanations(
    sample_size: int = 2000,
    save_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Generate global SHAP summary (beeswarm) and bar plots using a sample of the TEST set.

    Guarantees that all fraud rows in the test set are included in the sample.
    Does NOT use the training set.
    """
    target_dir = Path(save_dir) if save_dir else DOCS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    explainer = get_explainer()

    print("[1/3] Loading TEST split for global SHAP analysis...")
    df = load_data()
    _, _, test_df = time_based_split(df)

    fraud_df = test_df[test_df["Class"] == 1]
    non_fraud_df = test_df[test_df["Class"] == 0]

    num_fraud = len(fraud_df)
    needed_non_fraud = max(0, sample_size - num_fraud)
    non_fraud_sample = non_fraud_df.sample(n=min(needed_non_fraud, len(non_fraud_df)), random_state=SEED)

    sample_test_df = pd.concat([fraud_df, non_fraud_sample]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(
        f"  Created test sample of N={len(sample_test_df)} "
        f"(Frauds: {num_fraud}, Legitimate: {len(non_fraud_sample)})"
    )

    print("[2/3] Transforming test sample and computing SHAP values...")
    raw_feature_cols = ["Time", "Amount"] + PCA_FEATURES
    X_sample_scaled = explainer.preprocessor.transform(sample_test_df[raw_feature_cols])

    if explainer.is_hybrid:
        recon_err = explainer._compute_reconstruction_error(X_sample_scaled)
        X_sample = np.column_stack([X_sample_scaled, recon_err])
    else:
        X_sample = X_sample_scaled

    shap_values = explainer.explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    elif shap_values.ndim == 3:
        shap_values_pos = shap_values[:, :, 1]
    else:
        shap_values_pos = shap_values

    # Convert to DataFrame for readable plotting
    X_df = pd.DataFrame(X_sample, columns=explainer.feature_names)

    # 1. Beeswarm summary plot
    summary_plot_path = target_dir / "shap_summary.png"
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_values_pos,
        X_df,
        show=False,
        max_display=15,
    )
    plt.title("SHAP Global Summary (Test Set Sample)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved global summary plot -> {summary_plot_path}")

    # 2. Bar plot of mean absolute SHAP values
    bar_plot_path = target_dir / "shap_feature_importance.png"
    plt.figure(figsize=(10, 8), dpi=300)
    shap.summary_plot(
        shap_values_pos,
        X_df,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title("SHAP Mean |Value| Feature Importance (Test Set Sample)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved feature importance plot -> {bar_plot_path}")

    # Compute top 5 global features by mean absolute SHAP value
    mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)
    feat_importance = sorted(
        zip(explainer.feature_names, mean_abs_shap),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_5_global = [{"feature": f, "mean_abs_shap": float(val)} for f, val in feat_importance[:5]]

    print("\nTop 5 Global Features by Mean |SHAP|:")
    for rank, item in enumerate(top_5_global, 1):
        print(f"  {rank}. {item['feature']:<12} : {item['mean_abs_shap']:.4f}")

    return {
        "summary_plot_path": str(summary_plot_path),
        "bar_plot_path": str(bar_plot_path),
        "sample_size": len(sample_test_df),
        "fraud_count": num_fraud,
        "top_5_global": top_5_global,
    }


def benchmark_explain_latency(num_transactions: int = 100) -> dict[str, float]:
    """Time explain() calls over individual transactions and compute mean and p95 latency."""
    print(f"\n[3/3] Benchmarking explain() latency over {num_transactions} test transactions...")
    explainer = get_explainer()

    df = load_data()
    _, _, test_df = time_based_split(df)
    test_sample = test_df.sample(n=num_transactions, random_state=SEED)

    latencies_ms: list[float] = []

    # Warm-up run
    first_row = test_sample.iloc[0].to_dict()
    _ = explainer.explain(first_row)

    # Benchmark loop over single-transaction dicts
    for _, row in test_sample.iterrows():
        tx_dict = row.to_dict()
        start = time.perf_counter()
        _ = explainer.explain(tx_dict)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    latencies_arr = np.array(latencies_ms)
    avg_latency = float(np.mean(latencies_arr))
    p95_latency = float(np.percentile(latencies_arr, 95))
    max_latency = float(np.max(latencies_arr))

    print(f"  Average Latency: {avg_latency:.2f} ms")
    print(f"  P95 Latency:     {p95_latency:.2f} ms")
    print(f"  Max Latency:     {max_latency:.2f} ms")
    print(f"  SLA Check (<200 ms): {'PASS' if p95_latency < 200.0 else 'FAIL'}")

    return {
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "max_latency_ms": max_latency,
    }


def run_phase4() -> None:
    """Execute complete Phase 4 explainability pipeline."""
    print("=" * 60)
    print("PHASE 4: MODEL EXPLAINABILITY & SHAP BENCHMARKING")
    print("=" * 60)

    # 1. Global SHAP analysis
    generate_global_explanations(sample_size=2000)

    # 2. Latency benchmark
    benchmark_explain_latency(num_transactions=100)

    print("\nPhase 4 explainability run completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run_phase4()
