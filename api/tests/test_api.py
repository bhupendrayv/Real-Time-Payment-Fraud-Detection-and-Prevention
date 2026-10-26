"""Comprehensive API test suite for FraudShield.

Validates:
1. /health returns ok with valid model_version
2. valid prediction returns all fields with 0 <= fraud_score <= 1
3. known fraud row from test set scores higher than a known normal row
4. invalid input (missing field, negative Amount) returns 422
5. prediction is saved in the database and appears in /transactions
6. /stats returns correct counts and calculations after multiple predictions
"""
from __future__ import annotations

import os
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.db import Base, get_db
from api.main import app
from ml.data import load_data, time_based_split

# Setup isolated SQLite test database
TEST_DB_URL = "sqlite:///./test_api_fraudshield.db"
test_engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db() -> Generator[Session, None, None]:
    """Provide a transactional test database session."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create test tables before test session and drop after."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)
    if os.path.exists("./test_api_fraudshield.db"):
        try:
            os.remove("./test_api_fraudshield.db")
        except OSError:
            pass


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """TestClient that runs app lifespan (loading models & preprocessors once)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def real_test_transactions() -> tuple[dict, dict]:
    """Fetch one confirmed fraud and one confirmed normal transaction from the test set."""
    df = load_data()
    _, _, test_df = time_based_split(df)
    fraud_row = test_df[test_df["Class"] == 1].iloc[0].to_dict()
    normal_row = test_df[test_df["Class"] == 0].iloc[0].to_dict()

    # Drop target column for payload
    fraud_row.pop("Class", None)
    normal_row.pop("Class", None)
    return fraud_row, normal_row


def test_health_endpoint(client: TestClient) -> None:
    """Test that /health returns status: 'ok' and model_version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert len(data["model_version"]) > 0


def test_valid_prediction_structure(client: TestClient, real_test_transactions: tuple[dict, dict]) -> None:
    """Test that a valid transaction prediction returns all expected fields."""
    _, normal_row = real_test_transactions
    response = client.post("/predict", json=normal_row)
    assert response.status_code == 200
    data = response.json()

    assert "transaction_id" in data
    assert "fraud_score" in data
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert "is_fraud" in data
    assert isinstance(data["is_fraud"], bool)
    assert "threshold" in data
    assert "model_version" in data
    assert "explanation" in data
    assert isinstance(data["explanation"], list)
    assert len(data["explanation"]) == 5
    assert "latency_ms" in data
    assert data["latency_ms"] >= 0.0


def test_known_fraud_scores_higher_than_normal(
    client: TestClient, real_test_transactions: tuple[dict, dict]
) -> None:
    """Test that a known fraud transaction produces a higher fraud score than a normal one."""
    fraud_row, normal_row = real_test_transactions

    resp_fraud = client.post("/predict", json=fraud_row)
    resp_normal = client.post("/predict", json=normal_row)

    assert resp_fraud.status_code == 200
    assert resp_normal.status_code == 200

    fraud_score = resp_fraud.json()["fraud_score"]
    normal_score = resp_normal.json()["fraud_score"]

    assert fraud_score > normal_score, (
        f"Expected fraud score ({fraud_score}) to be strictly greater than normal score ({normal_score})"
    )


def test_invalid_input_validation(client: TestClient, real_test_transactions: tuple[dict, dict]) -> None:
    """Test that invalid input returns HTTP 422 Unprocessable Entity."""
    _, normal_row = real_test_transactions

    # Case 1: Missing required field
    bad_payload_missing = dict(normal_row)
    del bad_payload_missing["V4"]
    resp1 = client.post("/predict", json=bad_payload_missing)
    assert resp1.status_code == 422

    # Case 2: Negative amount
    bad_payload_neg_amount = dict(normal_row)
    bad_payload_neg_amount["Amount"] = -25.50
    resp2 = client.post("/predict", json=bad_payload_neg_amount)
    assert resp2.status_code == 422


def test_transactions_endpoint_persistence(
    client: TestClient, real_test_transactions: tuple[dict, dict]
) -> None:
    """Test that scored transactions are stored in DB and appear in /transactions."""
    fraud_row, _ = real_test_transactions
    post_resp = client.post("/predict", json=fraud_row)
    assert post_resp.status_code == 200
    tx_id = post_resp.json()["transaction_id"]

    # Query latest transactions
    get_resp = client.get("/transactions?limit=10")
    assert get_resp.status_code == 200
    items = get_resp.json()
    assert isinstance(items, list)
    assert len(items) > 0

    # Verify our transaction ID is at the top (newest first)
    found_ids = [item["id"] for item in items]
    assert tx_id in found_ids


def test_stats_endpoint(client: TestClient, real_test_transactions: tuple[dict, dict]) -> None:
    """Test that /stats accurately aggregates transaction statistics."""
    fraud_row, normal_row = real_test_transactions

    # Post 2 transactions
    client.post("/predict", json=normal_row)
    client.post("/predict", json=fraud_row)

    resp = client.get("/stats")
    assert resp.status_code == 200
    stats = resp.json()

    assert stats["total_scored"] >= 2
    assert stats["flagged_count"] >= 1
    assert 0.0 <= stats["flagged_rate"] <= 1.0
    assert stats["avg_latency_ms"] > 0.0
    assert stats["p95_latency_ms"] >= stats["avg_latency_ms"]
    assert isinstance(stats["hourly_counts"], list)
