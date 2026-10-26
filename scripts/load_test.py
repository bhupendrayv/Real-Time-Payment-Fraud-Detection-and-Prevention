"""Asynchronous Load Testing Script for FraudShield API.

Measures:
- Requests Per Second (RPS)
- p50, p95, p99, and mean latency
- Comparison between explain=True vs explain=False
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx
import numpy as np

from ml.data import load_data, time_based_split

API_BASE_URL = "http://127.0.0.1:8000"


async def send_prediction_request(
    client: httpx.AsyncClient, payload: dict[str, Any], explain: bool
) -> float:
    """Send a single transaction prediction request and return elapsed milliseconds."""
    start = time.perf_counter()
    resp = await client.post(f"{API_BASE_URL}/predict?explain={'true' if explain else 'false'}", json=payload)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed with status {resp.status_code}: {resp.text}")
    return elapsed_ms


async def run_load_test(
    concurrency: int = 10, total_requests: int = 100, explain: bool = True
) -> dict[str, float]:
    """Run concurrent load test against /predict."""
    df = load_data()
    _, _, test_df = time_based_split(df)
    test_rows = test_df.sample(n=min(total_requests, len(test_df)), random_state=42).to_dict(orient="records")
    for r in test_rows:
        r.pop("Class", None)

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for row in test_rows:
        queue.put_nowait(row)

    latencies_ms: list[float] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        async def worker():
            while not queue.empty():
                try:
                    payload = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                elapsed = await send_prediction_request(client, payload, explain)
                latencies_ms.append(elapsed)
                queue.task_done()

        start_time = time.perf_counter()
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        await asyncio.gather(*workers)
        total_wall_time = time.perf_counter() - start_time

    lat_arr = np.array(latencies_ms)
    rps = len(latencies_ms) / total_wall_time
    p50 = float(np.percentile(lat_arr, 50))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(np.mean(lat_arr))

    return {
        "requests": len(latencies_ms),
        "wall_time_s": total_wall_time,
        "rps": rps,
        "mean_latency_ms": mean_lat,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
    }


async def main():
    print("=" * 65)
    print("FRAUDSHIELD API LOAD TEST")
    print("=" * 65)

    # Health check
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{API_BASE_URL}/health")
            print(f"Health check: {resp.status_code} -> {resp.json()}")
        except Exception:
            print(f"Could not connect to {API_BASE_URL}. Ensure uvicorn server is running!")
            return

    # Scenario 1: With TreeSHAP explainability (explain=true)
    print("\n[Scenario 1] Running 100 requests with concurrency=10 (explain=True)...")
    res_explain = await run_load_test(concurrency=10, total_requests=100, explain=True)
    print(f"  RPS:          {res_explain['rps']:.1f} req/s")
    print(f"  Mean Latency: {res_explain['mean_latency_ms']:.2f} ms")
    print(f"  P50 Latency:  {res_explain['p50_latency_ms']:.2f} ms")
    print(f"  P95 Latency:  {res_explain['p95_latency_ms']:.2f} ms")
    print(f"  P99 Latency:  {res_explain['p99_latency_ms']:.2f} ms")

    # Scenario 2: High-throughput raw scoring (explain=false)
    print("\n[Scenario 2] Running 100 requests with concurrency=10 (explain=False)...")
    res_no_explain = await run_load_test(concurrency=10, total_requests=100, explain=False)
    print(f"  RPS:          {res_no_explain['rps']:.1f} req/s")
    print(f"  Mean Latency: {res_no_explain['mean_latency_ms']:.2f} ms")
    print(f"  P50 Latency:  {res_no_explain['p50_latency_ms']:.2f} ms")
    print(f"  P95 Latency:  {res_no_explain['p95_latency_ms']:.2f} ms")
    print(f"  P99 Latency:  {res_no_explain['p99_latency_ms']:.2f} ms")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
