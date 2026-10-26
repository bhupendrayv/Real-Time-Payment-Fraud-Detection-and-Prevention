"""Unit and integration tests for the Payment Stream Simulator."""
from __future__ import annotations

import json
import pytest
import httpx

from simulator.stream import TransactionSimulator


@pytest.fixture
def mock_api_transport():
    """Mock HTTPX Transport mimicking /health and /predict endpoints."""
    sent_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/health" in url_str:
            return httpx.Response(200, json={"status": "ok", "model_version": "xgb-v1"})

        if "/predict" in url_str:
            data = json.loads(request.content.decode("utf-8"))
            sent_payloads.append(data)
            # Deterministic response
            is_fraud = data.get("Amount", 0) > 1000.0 or "fraud" in str(data)
            return httpx.Response(
                200,
                json={
                    "transaction_id": "test-uuid",
                    "fraud_score": 0.85 if is_fraud else 0.01,
                    "is_fraud": is_fraud,
                    "threshold": 0.0445,
                    "model_version": "xgb-v1",
                    "explanation": [],
                    "latency_ms": 3.5,
                },
            )

        return httpx.Response(404, json={"detail": "Not found"})

    transport = httpx.MockTransport(handler)
    return transport, sent_payloads


@pytest.mark.asyncio
async def test_class_label_never_included_in_payload():
    """Verify that the ground truth 'Class' label is NEVER leaked into the API request payload."""
    sim = TransactionSimulator(rate=5.0, duration=1.0, fraud_boost=0.0)

    for _ in range(25):
        payload, true_label = sim.get_next_transaction()
        assert "Class" not in payload, "Ground truth 'Class' label must be removed before sending!"
        assert true_label in (0, 1), "Valid integer true label must be returned locally"
        assert "Time" in payload
        assert "Amount" in payload
        assert "V1" in payload
        assert "V28" in payload


@pytest.mark.asyncio
async def test_fraud_boost_samples_only_fraud_rows():
    """Verify that with --fraud-boost 1.0, every sampled transaction is guaranteed to be a true fraud."""
    sim = TransactionSimulator(rate=5.0, duration=1.0, fraud_boost=1.0, seed=123)

    for _ in range(20):
        payload, true_label = sim.get_next_transaction()
        assert true_label == 1, f"Expected true fraud (Class=1), got {true_label}"
        assert "Class" not in payload


@pytest.mark.asyncio
async def test_rate_controller_timing(mock_api_transport):
    """Verify the simulator dispatches approximately the requested transactions count over time."""
    transport, sent_payloads = mock_api_transport
    target_rate = 15.0
    duration = 2.0  # Should send ~30 transactions (±5)

    async with httpx.AsyncClient(transport=transport) as client:
        sim = TransactionSimulator(
            rate=target_rate,
            duration=duration,
            fraud_boost=0.0,
            client=client,
        )
        stats = await sim.run()

    expected_target = int(target_rate * duration)
    assert abs(stats.total_sent - expected_target) <= 6, (
        f"Expected around {expected_target} sent transactions, got {stats.total_sent}"
    )
    assert len(sent_payloads) == stats.total_sent


@pytest.mark.asyncio
async def test_api_errors_resilience():
    """Verify that API errors (500, timeouts) are handled gracefully without crashing the simulator."""
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Simulated Internal Server Error"})

    err_transport = httpx.MockTransport(error_handler)

    async with httpx.AsyncClient(transport=err_transport) as client:
        sim = TransactionSimulator(
            rate=10.0,
            duration=1.0,
            fraud_boost=0.0,
            client=client,
        )
        stats = await sim.run()

    # Simulator must complete cleanly
    assert stats.total_sent == 0
    assert stats.errors > 0, "Simulator should increment error count on failures without raising uncaught exceptions"
