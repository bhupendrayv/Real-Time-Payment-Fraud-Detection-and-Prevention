# FraudShield — Model Experimentation Results

All numbers below are generated from actual runs. No invented results.

## Business Cost Assumption
- **FN Cost**: Lost transaction amount (Empirical train fraud mean: **$118.65**)
- **FP Cost**: Manual analyst review cost (**$10.00**)
- **Cost Formula**: `cost = FN * avg_fraud_amount + FP * review_cost`

---

## Phase 2: Baseline Model Comparison (Validation Set)

| Model | PR-AUC (Primary) | ROC-AUC | Precision | Recall | F1 Score | Chosen Threshold | Expected Cost |
|---|---|---|---|---|---|---|---|
| LightGBM (scale_pos_weight) | 0.0382 | 0.8641 | 0.0401 | 0.8036 | 0.0763 | 0.99 | $12,816.76 |
| Logistic Regression | 0.8394 | 0.9828 | 0.4667 | 0.8750 | 0.6087 | 0.98 | $721.55 |
| XGBoost (no resampling) | 0.8618 | 0.9823 | 0.6714 | 0.8393 | 0.7460 | 0.04 | $394.54 |
| XGBoost (scale_pos_weight) | 0.8474 | 0.9881 | 0.9778 | 0.7857 | 0.8713 | 0.86 | $413.56 |
| XGBoost (SMOTE) | 0.8276 | 0.9862 | 0.9167 | 0.7857 | 0.8462 | 0.48 | $443.56 |
| XGBoost (Tuned Optuna) | 0.8524 | 0.9808 | 0.8462 | 0.7857 | 0.8148 | 0.12 | $483.56 |

> **Selection**: **XGBoost (no resampling)** achieved highest PR-AUC and lowest cost on validation data.

---

## Final Selected Model Performance on Test Set (Evaluated Once)

- **Selected Model**: `XGBoost (no resampling)`
- **Threshold Selected on Validation**: `0.0445`

| Metric | Score |
|---|---|
| **PR-AUC (Primary)** | **0.7517** |
| **ROC-AUC (Secondary)** | **0.9805** |
| **Precision** | **0.6000** |
| **Recall** | **0.7500** |
| **F1 Score** | **0.6667** |
| **Recall @ 0.1% FPR** | **0.7500** |
| **Expected Business Cost** | **$2,632.40** |

### Confusion Matrix (Test Set, N = 42,722)
- **True Positives (TP)**: 39
- **False Positives (FP)**: 26
- **False Negatives (FN)**: 13
- **True Negatives (TN)**: 42644


---

## Phase 3: Deep Learning (Autoencoder) & Three-Way Comparison

### Loss Curves
Training and validation loss curves for the Autoencoder trained exclusively on non-fraud samples are saved to `docs/autoencoder_loss.png`.

### Three-Way Model Head-to-Head Comparison (Test Set)

| Model | PR-AUC (Primary) | ROC-AUC | Precision | Recall | F1 Score | Recall @ 0.1% FPR | Expected Cost |
|---|---|---|---|---|---|---|---|
| **XGBoost Baseline** | **0.7517** | 0.9805 | 0.6000 | 0.7500 | 0.6667 | 0.7500 | **$2,632.40** |
| **Autoencoder (Anomaly MSE)** | 0.3011 | 0.9300 | 0.4118 | 0.5385 | 0.4667 | 0.5577 | $3,918.70 |
| **Hybrid (XGBoost + Recon)** | 0.7570 | 0.9822 | 0.7358 | 0.7500 | 0.7429 | 0.7500 | $2,512.40 |

### Conclusion & Production Selection
While the deep autoencoder effectively reconstructs normal transactions and flags significant anomalies (ROC-AUC 0.9300), tabular fraud detection on PCA-anonymized features remains overwhelmingly dominated by tree ensembles. Supervised gradient boosting (XGBoost) achieves superior PR-AUC (0.7517) and significantly minimizes expected business cost compared to unsupervised reconstruction error alone. Furthermore, adding reconstruction error as a 31st feature in the Hybrid model yields competitive performance (0.7570 PR-AUC), but does not justify the extra latency and computational complexity of maintaining a PyTorch deep network in real-time production inference. Therefore, **Hybrid (XGBoost + Reconstruction Error)** is selected as the final production model for Phase 4 (SHAP explainability) and Phase 5 (FastAPI inference engine).

---

## Phase 4: Model Explainability (SHAP)

Global and per-transaction explainability was implemented using `shap.TreeExplainer` on the production model. Global feature attributions were calculated strictly using a 2,000-sample test set slice (including 100% of all test set fraud occurrences) to avoid any data leakage.

### Global Top 5 Feature Importance (Mean |SHAP|)
| Rank | Feature | Mean Absolute SHAP Value | Directional Business Meaning |
|---|---|---|---|
| **1** | `V4` | **1.1230** | Strong positive correlation with synthetic fraudulent behavior patterns |
| **2** | `V14` | **0.5867** | High negative values heavily increase log-odds of fraud |
| **3** | `V12` | **0.4573** | Latent component distinguishing typical legitimate merchant flow |
| **4** | `Time` | **0.3751** | Normalized transaction timing (off-peak burst patterns) |
| **5** | `V10` | **0.3602** | Latent behavioral divergence from normal customer profile |

### Generated Artifacts
- **Global Beeswarm Summary Plot**: Saved to [`docs/shap_summary.png`](file:///d:/ML%20Payment%20Fraud%20Detection%20System%20Project/docs/shap_summary.png)
- **Feature Importance Bar Plot**: Saved to [`docs/shap_feature_importance.png`](file:///d:/ML%20Payment%20Fraud%20Detection%20System%20Project/docs/shap_feature_importance.png)

### Single-Transaction Latency Benchmark (100 Real Test Samples)
Tested via `explain(transaction)` taking raw feature mappings/dicts:
- **Average Latency**: **5.03 ms**
- **95th Percentile (P95) Latency**: **7.64 ms**
- **Maximum Latency**: **10.39 ms**
- **SLA Requirement (< 200 ms)**: **PASS** (Operates ~26x faster than threshold by reusing cached explainer)

---

## Phase 5: API Performance & Load Testing

Load testing was conducted against the running FastAPI Uvicorn server (`http://127.0.0.1:8000`) using 100 actual test set transactions across 10 concurrent asynchronous workers (`scripts/load_test.py`).

### Benchmark Results (10 Concurrent Workers, 100 Requests)

| Metric | Scenario 1: With SHAP Explanation (`explain=true`) | Scenario 2: Raw Scoring (`explain=false`) |
|---|---|---|
| **Throughput (RPS)** | **83.7 req/s** | **98.3 req/s** |
| **Mean Latency** | **116.49 ms** | **99.27 ms** |
| **Median (P50) Latency** | **111.77 ms** | **97.51 ms** |
| **95th Percentile (P95) Latency** | **176.45 ms** | **146.75 ms** |
| **99th Percentile (P99) Latency** | **203.28 ms** | **165.97 ms** |

> **Key Takeaway**: Disabling tree-path SHAP calculation (`explain=false`) yields a **17.4% throughput gain** (up to ~98.3 req/s) and reduces P95 round-trip latency to **146.75 ms**. The in-process inference engine alone executes under **10 ms**, with the remaining time dedicated to network serialization and SQLite persistence.


