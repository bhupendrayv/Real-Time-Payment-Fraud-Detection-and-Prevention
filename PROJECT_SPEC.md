# Real-Time Payment Fraud Detection System — Project Spec

## 1. Project Overview

**Name:** FraudShield (BHUPENDRA YADAV)

**One-line pitch:** An end-to-end, production-style system that scores payment transactions for fraud in under 100 ms, explains every decision, and shows live alerts on a dashboard.

**Why it exists:** Payment fraud is rare (about 0.1–2% of transactions) but costly. A useful system must handle heavy class imbalance, score in real time, choose thresholds by business cost, and explain its decisions.

**Honest framing (use this in README and interviews):** Built on public data with a simulated transaction stream. It mirrors how real fraud systems work, but it is not connected to real payments.

**Goals**
- Strong, correctly evaluated ML model (gradient boosting baseline + deep learning autoencoder)
- Real-time inference API with explainability
- Live dashboard
- Dockerized and deployed with a public link
- Tests, monitoring basics, clean documentation

**Non-goals:** real payment processing, real customer data, authentication for real users.

---

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 (ML + API), JavaScript/TypeScript (frontend) |
| Data/ML | pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, LightGBM |
| Deep learning | PyTorch (or TensorFlow 2/Keras — pick one and stay consistent) |
| Explainability | SHAP |
| Experiment tracking | MLflow (local) |
| API | FastAPI, Pydantic, Uvicorn |
| Database | PostgreSQL (or MongoDB) |
| Frontend | React + Vite, Tailwind CSS, Recharts |
| Containers | Docker, docker-compose |
| Testing | pytest, httpx |
| CI | GitHub Actions |
| Deploy | Render / Railway / Google Cloud Run (frontend on Vercel) |

---

## 3. Dataset

**Primary:** Kaggle "Credit Card Fraud Detection" (ULB): 284,807 transactions, 492 frauds, features V1–V28 (PCA-anonymized), Time, Amount, Class.

**Optional upgrade:** IEEE-CIS Fraud Detection (richer features, more realistic).

**Rules**
- Never commit raw data to Git. Add data/ to .gitignore. Add a scripts/download_data.py or README instructions.
- Split by time, not randomly. Train = earliest 70%, validation = next 15%, test = last 15%. A random split leaks future information.
- Fit scalers/encoders on train only.

---

## 4. Repository Structure

fraudshield/
├── PROJECT_SPEC.md
├── README.md
├── docker-compose.yml
├── .github/workflows/ci.yml
├── data/                     (gitignored)
├── notebooks/
│   └── 01_eda.ipynb
├── ml/
│   ├── config.py
│   ├── data.py               (loading, time-based split)
│   ├── features.py
│   ├── train_xgb.py
│   ├── train_autoencoder.py
│   ├── evaluate.py           (metrics, threshold selection, plots)
│   ├── explain.py            (SHAP)
│   └── artifacts/            (saved models)
├── api/
│   ├── Dockerfile
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── db.py
│   └── tests/
├── simulator/
│   └── stream.py             (replays test transactions to the API)
├── frontend/
│   ├── Dockerfile
│   └── src/ (components, pages, api client)
├── monitoring/
│   └── drift.py
└── docs/
    ├── architecture.png
    └── results.md

---

## 5. Phases (build in this order)

### Phase 1 — Setup and EDA
Tasks
- Create the repo structure, virtual environment, requirements.txt, .gitignore.
- Download the dataset, write ml/data.py with a time-based split function.
- EDA notebook: class balance, amount distribution by class, time patterns, missing values, feature correlations.

Acceptance criteria
- `python -m ml.data` prints split sizes and fraud rate per split.
- Each split has fraud cases.
- EDA notebook runs top to bottom with no errors.

### Phase 2 — Baseline ML model (XGBoost / LightGBM)
Tasks
- Train a logistic regression baseline first (sanity check).
- Train XGBoost with scale_pos_weight. Compare against: class weights, SMOTE (train only), and no resampling.
- Tune with Optuna or randomized search on the validation set (keep it modest).
- Log every run in MLflow (params, metrics, artifacts).

Metrics (do NOT use accuracy)
- PR-AUC (primary), ROC-AUC (secondary)
- Precision, recall, F1 at the chosen threshold
- Recall at fixed false-positive rate (e.g. 0.1% FPR)
- Confusion matrix
- Expected cost: cost = FN × avg_fraud_amount + FP × review_cost (state your assumed numbers)

Threshold selection: choose the threshold on the validation set that minimizes expected cost, then report on the test set once.

Acceptance criteria
- Results table comparing all variants, saved in docs/results.md.
- Final model + threshold saved to ml/artifacts/.

### Phase 3 — Deep learning (autoencoder)
Tasks
- Train an autoencoder (e.g. input → 64 → 32 → 16 → 32 → 64 → input) on non-fraud transactions only from the training set.
- Use reconstruction error as the anomaly score; choose threshold on validation.
- Compare with XGBoost on the same test metrics.
- Hybrid: add reconstruction error as a feature to XGBoost and retrain. Compare.

Acceptance criteria
- Training/validation loss curves saved.
- A three-way comparison (XGBoost vs autoencoder vs hybrid) with a one-paragraph honest conclusion.
- Pick the final production model based on the numbers.

### Phase 4 — Explainability
Tasks
- SHAP summary plot (global) and per-transaction explanation (top 5 contributing features).
- A function explain(transaction) -> list of {feature, contribution} usable by the API.

Acceptance criteria
- Explanations are returned in under 200 ms for a single transaction (use TreeExplainer).

### Phase 5 — Inference API (FastAPI)
Endpoints
- GET /health -> {status: "ok", model_version: "..."}
- POST /predict -> request: transaction features (validated with Pydantic). Response:
  {
    "transaction_id": "uuid",
    "fraud_score": 0.93,
    "is_fraud": true,
    "threshold": 0.41,
    "model_version": "xgb-v1",
    "explanation": [{"feature": "V14", "contribution": 1.8}],
    "latency_ms": 12
  }
- GET /transactions?limit=50&flagged_only=true
- GET /stats -> counts, flagged rate, avg latency, last 24h trend

Requirements
- Load the model once at startup, not per request.
- Store every scored transaction in the database.
- Input validation, clear error messages, request logging.
- p95 latency under 100 ms for /predict (measure and report it).

Acceptance criteria
- pytest tests for health, valid prediction, invalid input, and DB persistence.
- Load test (e.g. locust or hey) result recorded in docs/results.md.

### Phase 6 — Transaction simulator
- simulator/stream.py replays test-set transactions to /predict at a configurable rate (e.g. 5/sec), with an option to inject bursts of fraud.
- CLI flags: --rate, --duration, --fraud-boost.

### Phase 7 — Frontend dashboard (React)
Pages/components
- Live feed: table of recent transactions with score, status badge (Approved / Flagged), timestamp; auto-refresh every 2 seconds.
- Stat cards: total scored, flagged count, flagged %, avg latency.
- Chart: flagged transactions over time.
- Detail drawer: click a transaction to see its SHAP explanation as a horizontal bar chart.
- Manual test form: enter a transaction and get a score.

Acceptance criteria
- Responsive, loading and error states handled, no console errors.

### Phase 8 — Monitoring
- monitoring/drift.py: compare the distribution of recent incoming features vs training data (PSI or KS test). Log a warning above a threshold.
- Log model version with every prediction.
- Expose drift status in /stats.

### Phase 9 — Docker, CI, deployment
- Dockerfiles for API and frontend, docker-compose.yml for API + DB + frontend. `docker compose up` must work on a clean machine.
- GitHub Actions: lint (ruff), tests (pytest), build Docker images.
- Deploy API (Render/Railway/Cloud Run) and frontend (Vercel). Add the live URLs to the README.
- Use environment variables for secrets; never commit .env.

### Phase 10 — Documentation and polish
- README: problem, architecture diagram, results table, how to run locally, API examples, screenshots, demo GIF, limitations, future work.
- docs/results.md: all experiments and the final model's metrics.
- Clean commit history with meaningful messages.

---

## 6. Quality Rules (follow strictly)

1. No data leakage: time-based split, fit preprocessing on train only.
2. Never report accuracy as the main metric.
3. Set random seeds for reproducibility.
4. Type hints and docstrings on public functions.
5. Config in one place (ml/config.py), no magic numbers scattered around.
6. Every phase ends with a run and a short summary of what was verified.
7. Do not invent results. All numbers in docs must come from actual runs.
8. Keep dependencies pinned in requirements.txt.

---

## 7. Working Rules for the Agent

- Work one phase at a time, starting with Phase 1.
- Before writing code for a phase, give me a short plan and wait for my approval.
- After finishing a phase, run the code and tests, then summarize what you verified and show real outputs.
- If something in the spec is ambiguous or a dataset/library issue blocks you, tell me instead of guessing.
- Keep code simple, typed, tested, and readable. I must be able to explain every file in an interview.

---

## 8. Definition of Done

- All phases complete with acceptance criteria met
- `docker compose up` runs the full system locally
- Live deployed link works
- Tests pass in CI
- README has results, diagram, screenshots, demo GIF
- I can explain every design choice without looking at the code
