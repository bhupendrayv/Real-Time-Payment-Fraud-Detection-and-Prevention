"""Comprehensive test suite for FraudShield Drift Monitoring system.

Verifies:
1. PSI is approximately 0 on identical distributions and clearly > 0.25 on strongly shifted distributions.
2. Status mapping logic for the three operational thresholds (ok, warning, drift).
3. "insufficient_data" status is returned when rolling window is below min_rows.
4. GET /drift endpoint returns expected response contract and fields.
5. Injected synthetic drift in the database triggers "drift" status and correctly identifies the drifted feature.
"""
from __future__ import annotations

import datetime
import uuid
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.db import Base, TransactionRecord, get_db
from api.main import app
from ml.config import (
    DRIFT_FEATURES,
    PSI_OK_THRESHOLD,
    PSI_WARNING_THRESHOLD,
)
from monitoring.drift import (
    calculate_psi,
    drift_monitor,
    map_psi_to_status,
)

# Isolated test DB
TEST_DB_URL = "sqlite:///./test_monitoring_fraudshield.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="module")
def client():
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)
    with TestClient(app) as test_client:
        yield test_client
    Base.metadata.drop_all(bind=test_engine)
    app.dependency_overrides.clear()


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=test_engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=test_engine)


# ── Test 1: PSI identical vs shifted ─────────────────────────────────
def test_psi_identical_vs_shifted():
    """Verify PSI is close to 0 for identical distributions and > 0.25 for shifted distributions."""
    rng = np.random.default_rng(42)
    ref_data = rng.normal(loc=0.0, scale=1.0, size=5000)

    # 10 decile bins
    quantiles = np.linspace(0, 100, 11)
    raw_edges = np.percentile(ref_data, quantiles)
    edges = [-float("inf")] + list(raw_edges[1:-1]) + [float("inf")]
    props = [0.10] * 10

    # 1. Identical distribution
    identical_data = rng.normal(loc=0.0, scale=1.0, size=5000)
    psi_identical = calculate_psi(identical_data, edges, props)
    assert psi_identical < 0.05, f"PSI for identical data should be ~0, got {psi_identical}"

    # 2. Strongly shifted distribution (+2.5 sigma)
    shifted_data = rng.normal(loc=2.5, scale=1.0, size=5000)
    psi_shifted = calculate_psi(shifted_data, edges, props)
    assert psi_shifted > 0.25, f"PSI for shifted data should be > 0.25, got {psi_shifted}"


# ── Test 2: Status mapping for thresholds ────────────────────────────
def test_status_mapping_thresholds():
    """Verify mapping of PSI values to ok, warning, and drift."""
    assert map_psi_to_status(0.00) == "ok"
    assert map_psi_to_status(0.05) == "ok"
    assert map_psi_to_status(PSI_OK_THRESHOLD - 0.01) == "ok"

    assert map_psi_to_status(PSI_OK_THRESHOLD) == "warning"
    assert map_psi_to_status(0.18) == "warning"
    assert map_psi_to_status(PSI_WARNING_THRESHOLD) == "warning"

    assert map_psi_to_status(PSI_WARNING_THRESHOLD + 0.01) == "drift"
    assert map_psi_to_status(0.50) == "drift"
    assert map_psi_to_status(1.50) == "drift"


# ── Test 3: Insufficient data when window is too small ───────────────
def test_insufficient_data_when_window_too_small():
    """Verify that fewer than min_rows returns status 'insufficient_data'."""
    features = {feat: np.zeros(50) for feat in DRIFT_FEATURES}
    scores = np.zeros(50)

    res = drift_monitor.evaluate_features_and_scores(
        features_dict=features,
        fraud_scores=scores,
        min_rows=200,
    )

    assert res["status"] == "insufficient_data"
    assert res["window_size"] == 50
    assert res["min_rows_required"] == 200
    assert len(res["top_drifting_features"]) == 0


# ── Test 4: GET /drift returns expected fields ───────────────────────
def test_get_drift_endpoint_contract(client: TestClient):
    """Verify GET /drift responds with complete schema."""
    resp = client.get("/drift")
    assert resp.status_code == 200

    data = resp.json()
    assert "status" in data
    assert data["status"] in ("ok", "warning", "drift", "insufficient_data")
    assert "window_size" in data
    assert "computed_at" in data
    assert "score_psi" in data
    assert "flagged_rate" in data
    assert "reference_flagged_rate" in data
    assert "top_drifting_features" in data
    assert isinstance(data["top_drifting_features"], list)


# ── Test 5: Injected drift in DB produces drift status and flags feature ───
def test_injected_drift_in_db_produces_drift_status(db_session: Session):
    """Insert shifted V14 transactions and verify drift status & V14 top ranking."""
    v14_ref_std = drift_monitor.reference.get("features", {}).get("V14", {}).get("std", 1.0)
    drift_monitor.reference.get("features", {}).get("V14", {}).get("mean", 0.0)

    from ml.data import load_data, time_based_split

    df = load_data()
    _, _, test_df = time_based_split(df)
    sample_rows = test_df.iloc[:250].copy()

    records = []
    for _, row in sample_rows.iterrows():
        feat_dict = row.to_dict()
        feat_dict.pop("Class", None)
        # Heavily shift V14 by +6 standard deviations
        feat_dict["V14"] = float(feat_dict["V14"] + 6.0 * v14_ref_std)

        rec = TransactionRecord(
            id=str(uuid.uuid4()),
            created_at=datetime.datetime.now(datetime.timezone.utc),
            features=feat_dict,
            fraud_score=0.92,
            is_fraud=True,
            threshold=0.835,
            model_version="xgb_v1.0.0",
            explanation=[],
            latency_ms=10.0,
        )
        records.append(rec)

    db_session.bulk_save_objects(records)
    db_session.commit()

    # Evaluate directly from DB
    result = drift_monitor.evaluate_from_db(db_session, window_size=500, min_rows=200)

    assert result["status"] == "drift"
    assert len(result["top_drifting_features"]) > 0
    top_feature = result["top_drifting_features"][0]
    assert top_feature["feature"] == "V14"
    assert top_feature["status"] == "drift"
    assert top_feature["psi"] > 0.25
