"""Transaction stream replay simulator for FraudShield.

Replays transactions strictly from the TEST split to POST /predict,
mimicking real-time credit card payment streams with configurable rate
and synthetic fraud boost for live demonstrations.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
import pandas as pd

from ml.data import load_data, time_based_split


@dataclass
class SimulationStats:
    """Tracks live and aggregate simulation metrics."""

    total_sent: int = 0
    errors: int = 0
    flagged_count: int = 0
    true_frauds_sent: int = 0
    true_frauds_caught: int = 0  # True Positives
    false_alarms: int = 0        # False Positives
    latencies_ms: list[float] = field(default_factory=list)

    def record_success(self, true_label: int, is_flagged: bool, latency_ms: float) -> None:
        self.total_sent += 1
        self.latencies_ms.append(latency_ms)

        if true_label == 1:
            self.true_frauds_sent += 1
            if is_flagged:
                self.true_frauds_caught += 1
        else:
            if is_flagged:
                self.false_alarms += 1

        if is_flagged:
            self.flagged_count += 1

    def record_error(self) -> None:
        self.errors += 1


class TransactionSimulator:
    """Replays transactions from the test split with precise rate regulation."""

    def __init__(
        self,
        rate: float = 5.0,
        duration: float = 60.0,
        fraud_boost: float = 0.0,
        api_url: str = "http://localhost:8000",
        seed: int = 42,
        explain: bool = False,
        client: httpx.AsyncClient | None = None,
        drift_feature: str | None = None,
        drift_shift: float = 0.0,
        drift_amount_multiplier: float = 1.0,
        data_path: str | None = None,
    ) -> None:
        self.rate = float(rate)
        self.duration = float(duration)
        self.fraud_boost = max(0.0, min(1.0, float(fraud_boost)))
        self.api_url = api_url.rstrip("/")
        self.seed = seed
        self.explain = explain
        self.custom_client = client
        self.data_path = data_path

        # Drift injection settings (affects only memory payload sent to API)
        self.drift_feature = drift_feature
        self.drift_shift = float(drift_shift)
        self.drift_amount_multiplier = float(drift_amount_multiplier)
        self.drift_shift_val = 0.0

        if self.drift_feature and self.drift_shift != 0.0:
            # Look up reference std from reference_stats.json
            ref_std = 1.0
            from ml.config import REFERENCE_STATS_PATH
            if REFERENCE_STATS_PATH.exists():
                try:
                    import json
                    with open(REFERENCE_STATS_PATH, "r", encoding="utf-8") as f:
                        r_data = json.load(f)
                    ref_std = float(r_data.get("features", {}).get(self.drift_feature, {}).get("std", 1.0))
                except Exception:
                    ref_std = 1.0
            self.drift_shift_val = self.drift_shift * ref_std
            print(
                f"[DRIFT INJECTION ACTIVE] Feature: {self.drift_feature} | "
                f"Shift: +{self.drift_shift} std ({self.drift_shift_val:.4f})"
            )

        if self.drift_amount_multiplier != 1.0:
            print(f"[DRIFT INJECTION ACTIVE] Amount Multiplier: x{self.drift_amount_multiplier}")

        random.seed(self.seed)
        np.random.seed(self.seed)

        self.stats = SimulationStats()
        self.is_running = False

        # Load test split
        self._load_test_data()

    def _load_test_data(self) -> None:
        """Load test dataset without re-splitting, or load directly from specified data_path."""
        if self.data_path:
            from pathlib import Path
            p = Path(self.data_path)
            if not p.exists():
                raise FileNotFoundError(f"Specified data path does not exist: {p}")
            self.test_df = pd.read_csv(p).reset_index(drop=True)
            print(f"[SIMULATOR] Loaded {len(self.test_df)} transactions from custom path: {p}")
        else:
            df = load_data()
            _, _, test_df = time_based_split(df)
            self.test_df = test_df.reset_index(drop=True)

        self.fraud_indices = self.test_df[self.test_df["Class"] == 1].index.tolist()
        self.total_test_rows = len(self.test_df)
        self.current_index = 0

    def get_next_transaction(self) -> tuple[dict[str, Any], int]:
        """Fetch next transaction, applying fraud boost if requested.

        Returns:
            (payload_dict_without_Class, true_label)
        """
        # Determine whether to sample a guaranteed fraud transaction
        if self.fraud_boost > 0.0 and random.random() < self.fraud_boost and self.fraud_indices:
            idx = random.choice(self.fraud_indices)
        else:
            idx = self.current_index
            self.current_index = (self.current_index + 1) % self.total_test_rows

        row = self.test_df.iloc[idx].to_dict()
        true_label = int(row.pop("Class"))  # NEVER send Class label to API

        # Inject synthetic drift if requested (payload-only in memory)
        if self.drift_feature and self.drift_shift != 0.0 and self.drift_feature in row:
            row[self.drift_feature] = float(row[self.drift_feature] + self.drift_shift_val)

        if self.drift_amount_multiplier != 1.0 and "Amount" in row:
            row["Amount"] = max(0.0, float(row["Amount"] * self.drift_amount_multiplier))

        return row, true_label

    async def check_api_health(self, client: httpx.AsyncClient) -> bool:
        """Verify API /health endpoint is alive before streaming."""
        health_url = f"{self.api_url}/health"
        try:
            resp = await client.get(health_url, timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                print(f"[OK] Connected to API: {health_url} (Model: {data.get('model_version')})")
                return True
            print(f"[ERROR] API health check returned status {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            print(f"[ERROR] Could not connect to API at {health_url}: {e}")
            print("Please ensure the FastAPI server is running (`uvicorn api.main:app`).")
            return False

    async def send_transaction(
        self, client: httpx.AsyncClient, payload: dict[str, Any], true_label: int
    ) -> None:
        """Send single transaction to /predict with 1 retry on transient failure."""
        url = f"{self.api_url}/predict?explain={'true' if self.explain else 'false'}"
        for attempt in range(2):
            try:
                start_time = time.perf_counter()
                response = await client.post(url, json=payload, timeout=10.0)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                if response.status_code == 200:
                    data = response.json()
                    is_flagged = bool(data.get("is_fraud", False))
                    self.stats.record_success(true_label, is_flagged, latency_ms)
                    return
                else:
                    if attempt == 1:
                        self.stats.record_error()
                        # Silent error recording to avoid log spam, recorded in stats
            except Exception:
                if attempt == 1:
                    self.stats.record_error()
                await asyncio.sleep(0.05)

    async def run(self) -> SimulationStats:
        """Execute stream replay loop with drift-free interval timing."""
        self.is_running = True
        interval = 1.0 / self.rate

        client = self.custom_client if self.custom_client is not None else httpx.AsyncClient(timeout=15.0)

        try:
            # Health check before starting
            if self.custom_client is None:
                is_healthy = await self.check_api_health(client)
                if not is_healthy:
                    sys.exit(1)

            print(
                f"\n[STREAM START] Rate: {self.rate} tx/s | Duration: {self.duration}s | "
                f"Fraud Boost: {self.fraud_boost*100:.1f}%\n"
            )

            start_wall_time = time.perf_counter()
            next_log_time = start_wall_time + 5.0
            tx_count = 0

            while self.is_running:
                now = time.perf_counter()
                elapsed = now - start_wall_time
                if elapsed >= self.duration:
                    break

                # Periodic live progress log every 5s
                if now >= next_log_time:
                    recent_lats = self.stats.latencies_ms[-50:] if self.stats.latencies_ms else [0.0]
                    avg_lat = float(np.mean(recent_lats))
                    effective_rate = self.stats.total_sent / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{elapsed:5.1f}s] Sent: {self.stats.total_sent:4d} | "
                        f"Current Rate: {effective_rate:4.1f} tx/s | "
                        f"Flagged: {self.stats.flagged_count:2d} | "
                        f"Avg Latency: {avg_lat:5.1f} ms"
                    )
                    next_log_time = now + 5.0

                # Sample and dispatch
                payload, true_label = self.get_next_transaction()
                asyncio.create_task(self.send_transaction(client, payload, true_label))
                tx_count += 1

                # Accurate rate regulation preventing latency drift
                target_next_dispatch = start_wall_time + tx_count * interval
                sleep_needed = target_next_dispatch - time.perf_counter()
                if sleep_needed > 0:
                    await asyncio.sleep(sleep_needed)

            # Allow in-flight tasks to complete
            await asyncio.sleep(0.2)

        finally:
            self.is_running = False
            if self.custom_client is None:
                await client.aclose()

        self.print_summary()
        return self.stats

    def print_summary(self) -> None:
        """Display comprehensive final run metrics."""
        s = self.stats
        lats = np.array(s.latencies_ms) if s.latencies_ms else np.array([0.0])
        avg_lat = float(np.mean(lats))
        p95_lat = float(np.percentile(lats, 95)) if len(lats) > 1 else avg_lat

        precision = (
            (s.true_frauds_caught / s.flagged_count * 100) if s.flagged_count > 0 else 0.0
        )
        recall = (
            (s.true_frauds_caught / s.true_frauds_sent * 100) if s.true_frauds_sent > 0 else 0.0
        )

        print("\n" + "=" * 60)
        print("SIMULATION RUN SUMMARY")
        print("=" * 60)
        print(f"Total Transactions Sent:   {s.total_sent}")
        print(f"API Errors:                {s.errors}")
        print(f"Total Flagged as Fraud:    {s.flagged_count}")
        print(f"True Frauds Streamed:      {s.true_frauds_sent}")
        print(f"Frauds Caught (TP):        {s.true_frauds_caught}")
        print(f"False Alarms (FP):         {s.false_alarms}")
        print(f"Precision:                 {precision:.1f}%")
        print(f"Recall:                    {recall:.1f}%")
        print(f"Average Roundtrip Latency: {avg_lat:.2f} ms")
        print(f"P95 Latency:               {p95_lat:.2f} ms")
        print("=" * 60 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FraudShield Real-time Payment Stream Simulator")
    parser.add_argument("--rate", type=float, default=5.0, help="Transactions per second (default 5)")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default 60)")
    parser.add_argument(
        "--fraud-boost",
        type=float,
        default=0.0,
        help="Probability (0.0 to 1.0) of injecting a true fraud row (default 0.0)",
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Enable SHAP explanations in predict request (default false for high throughput)",
    )
    # Drift testing flags
    parser.add_argument(
        "--drift-feature",
        type=str,
        default=None,
        help="Feature name to inject synthetic drift into (e.g. V14)",
    )
    parser.add_argument(
        "--drift-shift",
        type=float,
        default=0.0,
        help="Number of reference standard deviations to shift drift-feature (e.g. 3.0)",
    )
    parser.add_argument(
        "--drift-amount-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to transaction Amount (e.g. 5.0)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to custom CSV (e.g. data/sample/demo_transactions.csv) instead of full dataset",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    sim = TransactionSimulator(
        rate=args.rate,
        duration=args.duration,
        fraud_boost=args.fraud_boost,
        api_url=args.api_url,
        seed=args.seed,
        explain=args.explain,
        drift_feature=args.drift_feature,
        drift_shift=args.drift_shift,
        drift_amount_multiplier=args.drift_amount_multiplier,
        data_path=args.data_path,
    )

    loop = asyncio.get_running_loop()

    def handle_signal():
        print("\n[STOPPING] Interrupted by user (Ctrl+C). Printing summary...")
        sim.is_running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # Signal handling on Windows when loop is Proactor
            pass

    try:
        await sim.run()
    except KeyboardInterrupt:
        sim.is_running = False
        sim.print_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
