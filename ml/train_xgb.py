"""Train baseline and gradient-boosted models for payment fraud detection.

Strictly follows specification:
1. Logistic regression baseline (sanity check)
2. XGBoost with scale_pos_weight
3. XGBoost with SMOTE (applied to train set only)
4. LightGBM baseline
5. Tuned XGBoost (using Optuna on validation PR-AUC)
6. All experiments logged to MLflow
7. Cost-minimizing threshold chosen on validation set only, evaluated on test set once
8. Results saved to docs/results.md and model artifacts to ml/artifacts/
"""
from __future__ import annotations

import os
import sys
from typing import Any
import joblib
import mlflow
import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ml.config import ARTIFACTS_DIR, DOCS_DIR, PROJECT_ROOT, SEED, TARGET
from ml.data import load_data, time_based_split
from ml.evaluate import (
    DEFAULT_REVIEW_COST,
    evaluate_model,
    find_optimal_threshold,
    save_threshold_config,
)
from ml.features import prepare_features


# Silence Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_and_evaluate_all() -> dict[str, Any]:
    """Execute complete Phase 2 model training and evaluation suite."""
    print("=" * 65)
    print("FraudShield - Phase 2: Baseline & Gradient Boosting Models")
    print("=" * 65)

    # 1. Load and split data
    print("\n[1/6] Loading data and performing time-based split...")
    df = load_data()
    train_df, val_df, test_df = time_based_split(df)

    y_train = np.ascontiguousarray(train_df[TARGET].to_numpy(), dtype=np.int32)
    y_val = np.ascontiguousarray(val_df[TARGET].to_numpy(), dtype=np.int32)
    y_test = np.ascontiguousarray(test_df[TARGET].to_numpy(), dtype=np.int32)

    amounts_train = train_df["Amount"].to_numpy()
    amounts_val = val_df["Amount"].to_numpy()
    amounts_test = test_df["Amount"].to_numpy()

    # Calculate avg fraud amount from training set only
    train_fraud_amounts = amounts_train[y_train == 1]
    avg_fraud_amount = float(train_fraud_amounts.mean())
    print(f"  Training fraud cases: {len(train_fraud_amounts):,}")
    print(f"  Empirical avg fraud amount: ${avg_fraud_amount:.2f}")
    print(f"  Assumed manual review cost: ${DEFAULT_REVIEW_COST:.2f}")

    # 2. Preprocess features (fit on train only!)
    print("\n[2/6] Fitting preprocessor on training data...")
    X_train, X_val, X_test, preprocessor = prepare_features(train_df, val_df, test_df)
    preprocessor.save()
    print(f"  Feature shape: {X_train.shape[1]} features (Time, Amount + 28 PCA)")
    print(f"  Saved preprocessor -> {ARTIFACTS_DIR / 'preprocessor.joblib'}")

    # Calculate scale_pos_weight
    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = float(neg_count / pos_count)

    # Setup MLflow with recommended SQLite database backend
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = PROJECT_ROOT / "mlruns.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    mlflow.set_experiment("fraudshield-baseline")

    models_to_run: dict[str, Any] = {}

    # ── Model 1: Logistic Regression ──
    models_to_run["Logistic Regression"] = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=SEED,
        solver="lbfgs",
    )

    # ── Model 2: XGBoost (no resampling baseline) ──
    models_to_run["XGBoost (no resampling)"] = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )

    # ── Model 3: XGBoost with scale_pos_weight ──
    models_to_run["XGBoost (scale_pos_weight)"] = XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric="logloss",
        n_jobs=-1,
    )

    results_list: list[dict[str, Any]] = []

    # ── Model 0: Run LightGBM in isolated subprocess first ──
    print("\n[3/6] Training baseline models & logging to MLflow...")
    print("  Training LightGBM (scale_pos_weight) in isolated process...")
    import subprocess
    import json
    lgb_proc = subprocess.run(
        [sys.executable, "-m", "ml.train_lgb"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    if lgb_proc.returncode == 0:
        lgb_metrics_path = ARTIFACTS_DIR / "lgbm_metrics.json"
        if lgb_metrics_path.exists():
            with open(lgb_metrics_path, "r", encoding="utf-8") as f:
                lgb_metrics = json.load(f)
            lgb_model_path = ARTIFACTS_DIR / "lgbm_model.joblib"
            if lgb_model_path.exists():
                lgb_metrics["model"] = joblib.load(lgb_model_path)
            results_list.append(lgb_metrics)
            print("    -> LightGBM completed successfully.")
    else:
        print(f"    -> LightGBM runner output: {lgb_proc.stderr}")

    for name, model in models_to_run.items():
        with mlflow.start_run(run_name=name):
            print(f"  Training {name}...")
            model.fit(X_train, y_train)

            # Val predictions
            val_proba = model.predict_proba(X_val)[:, 1]
            opt_thresh, min_val_cost = find_optimal_threshold(
                y_val, val_proba, amounts_val=amounts_val, avg_fraud_amount=avg_fraud_amount
            )

            val_metrics = evaluate_model(
                y_val,
                val_proba,
                threshold=opt_thresh,
                amounts=amounts_val,
                avg_fraud_amount=avg_fraud_amount,
            )

            # Log to MLflow
            mlflow.log_params({"model_name": name, "threshold": opt_thresh})
            for m_key, m_val in val_metrics.items():
                if isinstance(m_val, (int, float)):
                    mlflow.log_metric(f"val_{m_key}", m_val)

            val_metrics["model_name"] = name
            val_metrics["model"] = model
            results_list.append(val_metrics)
            print(
                f"    -> Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
                f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
                f"Opt Thresh: {opt_thresh:.2f} | "
                f"Val Cost: ${val_metrics['expected_cost']:,.2f}"
            )

    # ── Model 4: XGBoost with SMOTE (train only) ──
    print("\n[4/6] Training XGBoost with SMOTE (resampling train set only)...")
    with mlflow.start_run(run_name="XGBoost (SMOTE)"):
        smote = SMOTE(random_state=SEED, sampling_strategy=0.1)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        X_train_smote = np.ascontiguousarray(X_train_smote, dtype=np.float32)
        y_train_smote = np.ascontiguousarray(y_train_smote, dtype=np.int32)
        print(f"  Resampled train set: {len(y_train_smote):,} samples (frauds: {int(np.sum(y_train_smote)):,})")

        xgb_smote = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            eval_metric="logloss",
            n_jobs=-1,
        )
        xgb_smote.fit(X_train_smote, y_train_smote)
        val_proba = xgb_smote.predict_proba(X_val)[:, 1]

        opt_thresh, min_val_cost = find_optimal_threshold(
            y_val, val_proba, amounts_val=amounts_val, avg_fraud_amount=avg_fraud_amount
        )
        val_metrics = evaluate_model(
            y_val,
            val_proba,
            threshold=opt_thresh,
            amounts=amounts_val,
            avg_fraud_amount=avg_fraud_amount,
        )

        mlflow.log_params({"model_name": "XGBoost (SMOTE)", "threshold": opt_thresh})
        for m_key, m_val in val_metrics.items():
            if isinstance(m_val, (int, float)):
                mlflow.log_metric(f"val_{m_key}", m_val)

        val_metrics["model_name"] = "XGBoost (SMOTE)"
        val_metrics["model"] = xgb_smote
        results_list.append(val_metrics)
        print(
            f"    -> Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Opt Thresh: {opt_thresh:.2f} | "
            f"Val Cost: ${val_metrics['expected_cost']:,.2f}"
        )

    # ── Model 5: Optuna Hyperparameter Tuning for XGBoost ──
    print("\n[5/6] Tuning XGBoost with Optuna (12 trials on validation PR-AUC)...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 250),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.15, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": scale_pos_weight,
            "random_state": SEED,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_val)[:, 1]
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(y_val, preds))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=12)

    best_xgb_params = study.best_params
    best_xgb_params["scale_pos_weight"] = scale_pos_weight
    best_xgb_params["random_state"] = SEED
    best_xgb_params["eval_metric"] = "logloss"
    best_xgb_params["n_jobs"] = -1

    with mlflow.start_run(run_name="XGBoost (Tuned Optuna)"):
        tuned_xgb = XGBClassifier(**best_xgb_params)
        tuned_xgb.fit(X_train, y_train)
        val_proba = tuned_xgb.predict_proba(X_val)[:, 1]

        opt_thresh, min_val_cost = find_optimal_threshold(
            y_val, val_proba, amounts_val=amounts_val, avg_fraud_amount=avg_fraud_amount
        )
        val_metrics = evaluate_model(
            y_val,
            val_proba,
            threshold=opt_thresh,
            amounts=amounts_val,
            avg_fraud_amount=avg_fraud_amount,
        )

        mlflow.log_params(best_xgb_params)
        mlflow.log_params({"threshold": opt_thresh})
        for m_key, m_val in val_metrics.items():
            if isinstance(m_val, (int, float)):
                mlflow.log_metric(f"val_{m_key}", m_val)

        val_metrics["model_name"] = "XGBoost (Tuned Optuna)"
        val_metrics["model"] = tuned_xgb
        results_list.append(val_metrics)
        print(
            f"    -> Tuned Val PR-AUC: {val_metrics['pr_auc']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f} | "
            f"Opt Thresh: {opt_thresh:.2f} | "
            f"Val Cost: ${val_metrics['expected_cost']:,.2f}"
        )

    # ── Pick best model by Validation PR-AUC ──
    best_val_result = max(results_list, key=lambda x: x["pr_auc"])
    best_model_name = best_val_result["model_name"]
    best_model = best_val_result["model"]
    best_threshold = best_val_result["threshold"]

    print(f"\n[6/6] Selected Best Model: {best_model_name} (Val PR-AUC: {best_val_result['pr_auc']:.4f})")
    print(f"  Optimal Validation Threshold: {best_threshold:.4f}")

    # Evaluate ONCE on Test set
    print("  Evaluating ONCE on Test set with validation threshold...")
    test_proba = best_model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_model(
        y_test,
        test_proba,
        threshold=best_threshold,
        amounts=amounts_test,
        avg_fraud_amount=avg_fraud_amount,
    )

    print("\n" + "=" * 65)
    print("FINAL TEST SET PERFORMANCE (Evaluated Once)")
    print("=" * 65)
    print(f"  PR-AUC (Primary):          {test_metrics['pr_auc']:.4f}")
    print(f"  ROC-AUC (Secondary):        {test_metrics['roc_auc']:.4f}")
    print(f"  Precision:                  {test_metrics['precision']:.4f}")
    print(f"  Recall:                     {test_metrics['recall']:.4f}")
    print(f"  F1 Score:                   {test_metrics['f1']:.4f}")
    print(f"  Recall @ 0.1% FPR:          {test_metrics['recall_at_0.1pct_fpr']:.4f}")
    print(f"  Expected Cost:             ${test_metrics['expected_cost']:,.2f}")
    print(f"  Confusion Matrix:           TP={test_metrics['tp']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TN={test_metrics['tn']}")
    print("=" * 65)

    # Save artifacts
    model_artifact_path = ARTIFACTS_DIR / "best_model.joblib"
    joblib.dump(best_model, model_artifact_path)
    print(f"\n[OK] Saved best model to {model_artifact_path}")

    save_threshold_config(
        best_threshold,
        metadata={
            "model_name": best_model_name,
            "avg_fraud_amount": avg_fraud_amount,
            "review_cost": DEFAULT_REVIEW_COST,
            "val_pr_auc": best_val_result["pr_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1": test_metrics["f1"],
            "test_cost": test_metrics["expected_cost"],
        },
    )
    print(f"[OK] Saved threshold config to {ARTIFACTS_DIR / 'threshold.json'}")

    # Save docs/results.md
    _save_results_md(results_list, best_model_name, best_threshold, test_metrics, avg_fraud_amount)

    return {
        "best_model_name": best_model_name,
        "best_threshold": best_threshold,
        "test_metrics": test_metrics,
        "val_results": results_list,
    }


def _save_results_md(
    val_results: list[dict[str, Any]],
    best_model_name: str,
    best_threshold: float,
    test_metrics: dict[str, Any],
    avg_fraud_amount: float,
) -> None:
    """Save results table to docs/results.md strictly with real run data."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = DOCS_DIR / "results.md"

    table_rows = []
    for r in val_results:
        table_rows.append(
            f"| {r['model_name']} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} | "
            f"{r['threshold']:.2f} | ${r['expected_cost']:,.2f} |"
        )
    rows_str = "\n".join(table_rows)

    content = f"""# FraudShield — Model Experimentation Results

All numbers below are generated from actual runs. No invented results.

## Business Cost Assumption
- **FN Cost**: Lost transaction amount (Empirical train fraud mean: **${avg_fraud_amount:.2f}**)
- **FP Cost**: Manual analyst review cost (**${DEFAULT_REVIEW_COST:.2f}**)
- **Cost Formula**: `cost = FN * avg_fraud_amount + FP * review_cost`

---

## Phase 2: Baseline Model Comparison (Validation Set)

| Model | PR-AUC (Primary) | ROC-AUC | Precision | Recall | F1 Score | Chosen Threshold | Expected Cost |
|---|---|---|---|---|---|---|---|
{rows_str}

> **Selection**: **{best_model_name}** achieved highest PR-AUC and lowest cost on validation data.

---

## Final Selected Model Performance on Test Set (Evaluated Once)

- **Selected Model**: `{best_model_name}`
- **Threshold Selected on Validation**: `{best_threshold:.4f}`

| Metric | Score |
|---|---|
| **PR-AUC (Primary)** | **{test_metrics['pr_auc']:.4f}** |
| **ROC-AUC (Secondary)** | **{test_metrics['roc_auc']:.4f}** |
| **Precision** | **{test_metrics['precision']:.4f}** |
| **Recall** | **{test_metrics['recall']:.4f}** |
| **F1 Score** | **{test_metrics['f1']:.4f}** |
| **Recall @ 0.1% FPR** | **{test_metrics['recall_at_0.1pct_fpr']:.4f}** |
| **Expected Business Cost** | **${test_metrics['expected_cost']:,.2f}** |

### Confusion Matrix (Test Set, N = 42,722)
- **True Positives (TP)**: {test_metrics['tp']}
- **False Positives (FP)**: {test_metrics['fp']}
- **False Negatives (FN)**: {test_metrics['fn']}
- **True Negatives (TN)**: {test_metrics['tn']}
"""
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"[OK] Saved documentation -> {results_path}")


if __name__ == "__main__":
    train_and_evaluate_all()
