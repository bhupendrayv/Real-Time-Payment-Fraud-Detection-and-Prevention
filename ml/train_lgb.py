"""Isolated worker to train LightGBM without OpenMP collision with XGBoost on Windows."""
from __future__ import annotations

import json
import os
import joblib
import lightgbm as lgb
import mlflow
import numpy as np

from ml.config import ARTIFACTS_DIR, PROJECT_ROOT, SEED, TARGET
from ml.data import load_data, time_based_split
from ml.evaluate import evaluate_model, find_optimal_threshold
from ml.features import prepare_features


def run_lightgbm() -> dict:
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = PROJECT_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("fraudshield-baseline")

    df = load_data()
    train_df, val_df, test_df = time_based_split(df)
    X_train, X_val, X_test, _ = prepare_features(train_df, val_df, test_df)

    y_train = np.ascontiguousarray(train_df[TARGET].values, dtype=np.int32)
    y_val = np.ascontiguousarray(val_df[TARGET].values, dtype=np.int32)

    amounts_train = train_df["Amount"].to_numpy()
    amounts_val = val_df["Amount"].to_numpy()
    avg_fraud_amount = float(amounts_train[y_train == 1].mean())

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / pos_count)

    clf = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        verbosity=-1,
        n_jobs=1,
    )

    with mlflow.start_run(run_name="LightGBM (scale_pos_weight)"):
        clf.fit(X_train, y_train)
        val_proba = clf.predict_proba(X_val)[:, 1]

        opt_thresh, min_cost = find_optimal_threshold(
            y_val, val_proba, amounts_val=amounts_val, avg_fraud_amount=avg_fraud_amount
        )
        val_metrics = evaluate_model(
            y_val, val_proba, threshold=opt_thresh, amounts=amounts_val, avg_fraud_amount=avg_fraud_amount
        )

        mlflow.log_params({"model_name": "LightGBM (scale_pos_weight)", "threshold": opt_thresh})
        for m_key, m_val in val_metrics.items():
            if isinstance(m_val, (int, float)):
                mlflow.log_metric(f"val_{m_key}", m_val)

        val_metrics["model_name"] = "LightGBM (scale_pos_weight)"
        lgb_model_path = ARTIFACTS_DIR / "lgbm_model.joblib"
        joblib.dump(clf, lgb_model_path)
        val_metrics["model_artifact"] = str(lgb_model_path)

    metrics_out = ARTIFACTS_DIR / "lgbm_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    print(
        f"    -> Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
        f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
        f"Opt Thresh: {opt_thresh:.2f} | "
        f"Val Cost: ${val_metrics['expected_cost']:,.2f}"
    )
    return val_metrics


if __name__ == "__main__":
    run_lightgbm()
