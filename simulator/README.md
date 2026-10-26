# FraudShield — Transaction Stream Simulator

Replays transactions from the **test split** to the live FraudShield FastAPI, mimicking a real payment stream with configurable rate, duration, and fraud burst injection.

---

## Quick Start

Ensure the API is running first:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then run the simulator:
```bash
# Normal replay at natural fraud rate (5 tx/s for 60s)
python -m simulator.stream --rate 5 --duration 60

# Fraud burst demo (10 tx/s, 30% chance each tx is a real fraud row)
python -m simulator.stream --rate 10 --duration 30 --fraud-boost 0.3
```

---

## CLI Flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--rate` | float | `5` | Transactions per second |
| `--duration` | float | `60` | How long to run (seconds) |
| `--fraud-boost` | float | `0.0` | Probability (0.0–1.0) of injecting a real fraud row on each tick |
| `--api-url` | string | `http://localhost:8000` | FastAPI base URL |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--drift-feature` | string | `None` | Feature name to inject synthetic shift into (e.g. `V14`) |
| `--drift-shift` | float | `0.0` | Number of baseline standard deviations to shift `drift-feature` |
| `--drift-amount-multiplier` | float | `1.0` | Multiplier applied to transaction `Amount` |

---

## Drift Testing Examples

Inject synthetic data drift to verify the `/drift` monitoring pipeline:

```bash
# Shift feature V14 by +3 standard deviations
python -m simulator.stream --rate 10 --duration 60 --drift-feature V14 --drift-shift 3.0

# Inflate transaction amounts by 5x to trigger Amount drift
python -m simulator.stream --rate 10 --duration 60 --drift-amount-multiplier 5.0
```


## Live Console Output

Every 5 seconds:
```
[  5.0s] Sent:   50 | Current Rate: 10.0 tx/s | Flagged: 11 | Avg Latency:  15.8 ms
[ 10.0s] Sent:  100 | Current Rate: 10.0 tx/s | Flagged: 24 | Avg Latency:  16.4 ms
```

## Final Summary

```
============================================================
SIMULATION RUN SUMMARY
============================================================
Total Transactions Sent:   301
API Errors:                0
Total Flagged as Fraud:    73
True Frauds Streamed:      100
Frauds Caught (TP):        73
False Alarms (FP):         0
Precision:                 100.0%
Recall:                    73.0%
Average Roundtrip Latency: 16.11 ms
P95 Latency:               18.59 ms
============================================================
```

---

## Measured Results

### Scenario 1 — Normal Replay (`--rate 5 --duration 30`)
- **Sent**: 150 transactions, **Rate**: 5.0 tx/s (exact)
- **Flagged**: 0 (natural fraud rate — no fraud rows in first 150 test rows)
- **Avg Latency**: 15.75 ms | **P95**: 17.24 ms | **Errors**: 0

### Scenario 2 — Fraud Burst Demo (`--rate 10 --duration 30 --fraud-boost 0.3`)
- **Sent**: 301 transactions, **Rate**: 10.0 tx/s (exact)
- **True Frauds Injected**: 100 | **Flagged**: 73 | **False Alarms**: 0
- **Precision**: 100.0% | **Recall**: 73.0%
- **Avg Latency**: 16.11 ms | **P95**: 18.59 ms | **Errors**: 0

---

## Running Tests

```bash
python -m pytest simulator/tests/test_stream.py -v
```

Tests verify:
1. `Class` label is **never** included in API payloads
2. With `--fraud-boost 1.0`, every sampled row is a real fraud
3. Rate controller dispatches approximately the correct transaction count
4. API errors (500, timeout) do not crash the simulator
