"""FastAPI Application for FraudShield Payment Fraud Detection System."""
from __future__ import annotations

import datetime
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.config import CORS_ORIGINS, RATE_LIMIT_PER_MINUTE
from api.db import TransactionRecord, get_db, init_db
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from api.model_loader import model_manager
from api.schemas import (
    DriftResponse,
    DriftSummary,
    HealthResponse,
    HourlyCount,
    PredictResponse,
    StatsResponse,
    TransactionInput,
    TransactionItem,
)

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fraudshield.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan to initialize database and load model artifacts once at startup."""
    logger.info("Initializing database...")
    init_db()
    logger.info("Loading model, preprocessor, and SHAP explainer into memory...")
    model_manager.load()
    logger.info(
        f"FraudShield API successfully initialized. Model: {model_manager.model_version}, "
        f"Threshold: {model_manager.threshold:.4f}, Hybrid: {model_manager.is_hybrid}"
    )
    yield
    logger.info("Shutting down FraudShield API...")


app = FastAPI(
    title="FraudShield Inference API",
    description="Real-Time Payment Fraud Detection System with SHAP Explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# Setup rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log incoming requests with execution duration."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        f"{request.method} {request.url.path} - Status: {response.status_code} ({duration_ms:.2f}ms)"
    )
    return response


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Ensure internal errors do not leak stack traces to clients."""
    logger.exception(f"Unhandled error processing {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact system administrator."},
    )


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    """Return health status and loaded model version."""
    return HealthResponse(
        status="ok",
        model_version=model_manager.model_version,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
@limiter.limit(RATE_LIMIT_PER_MINUTE)
def predict(
    request: Request,
    payload: TransactionInput,
    explain: bool = Query(
        default=True,
        description="Whether to compute top 5 SHAP feature contributions",
    ),
    db: Session = Depends(get_db),
) -> PredictResponse:
    """Score a transaction for fraudulent activity, store in database, and return explanations."""
    tx_id = str(uuid.uuid4())
    tx_dict = payload.model_dump()

    start_time = time.perf_counter()
    try:
        fraud_score, is_fraud, explanations = model_manager.predict(
            tx_dict,
            explain=explain,
            top_k=5,
        )
    except Exception as e:
        logger.error(f"Inference error for transaction {tx_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing failed: {str(e)}",
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Store in database
    try:
        record = TransactionRecord(
            id=tx_id,
            created_at=datetime.datetime.now(datetime.timezone.utc),
            features=tx_dict,
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            threshold=model_manager.threshold,
            model_version=model_manager.model_version,
            explanation=explanations,
            latency_ms=latency_ms,
        )
        db.add(record)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist transaction {tx_id}: {e}")
        # Note: Do not fail inference response if DB logging encounters an issue, but log warning

    return PredictResponse(
        transaction_id=tx_id,
        fraud_score=round(fraud_score, 4),
        is_fraud=is_fraud,
        threshold=round(model_manager.threshold, 4),
        model_version=model_manager.model_version,
        explanation=explanations,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/transactions", response_model=list[TransactionItem], tags=["Audit"])
def get_transactions(
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return"),
    flagged_only: bool = Query(default=False, description="Filter only flagged fraud transactions"),
    db: Session = Depends(get_db),
) -> list[TransactionItem]:
    """Retrieve scored transactions, ordered newest first."""
    query = db.query(TransactionRecord)
    if flagged_only:
        query = query.filter(TransactionRecord.is_fraud.is_(True))

    records = query.order_by(TransactionRecord.created_at.desc()).limit(limit).all()
    return records


@app.get("/stats", response_model=StatsResponse, tags=["Monitoring"])
def get_stats(db: Session = Depends(get_db)) -> StatsResponse:
    """Return operational metrics including total scored, flagged rate, and latency stats."""
    total_scored = db.query(TransactionRecord).count()
    if total_scored == 0:
        return StatsResponse(
            total_scored=0,
            flagged_count=0,
            flagged_rate=0.0,
            avg_latency_ms=0.0,
            p95_latency_ms=0.0,
            hourly_counts=[],
        )

    flagged_count = db.query(TransactionRecord).filter(TransactionRecord.is_fraud.is_(True)).count()
    flagged_rate = float(flagged_count / total_scored)

    # Calculate average and p95 latency
    latencies = [row[0] for row in db.query(TransactionRecord.latency_ms).all()]
    avg_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Hourly counts over the last 24 hours
    now = datetime.datetime.now(datetime.timezone.utc)
    twenty_four_hours_ago = now - datetime.timedelta(hours=24)

    recent_txs = (
        db.query(TransactionRecord)
        .filter(TransactionRecord.created_at >= twenty_four_hours_ago)
        .all()
    )

    # Aggregate by hour bucket
    hour_buckets: dict[str, dict[str, int]] = {}
    for tx in recent_txs:
        # Format: YYYY-MM-DD HH:00
        hour_key = tx.created_at.strftime("%Y-%m-%d %H:00")
        if hour_key not in hour_buckets:
            hour_buckets[hour_key] = {"total": 0, "flagged": 0}
        hour_buckets[hour_key]["total"] += 1
        if tx.is_fraud:
            hour_buckets[hour_key]["flagged"] += 1

    sorted_hourly = [
        HourlyCount(hour=h, total=counts["total"], flagged=counts["flagged"])
        for h, counts in sorted(hour_buckets.items())
    ]

    # Model version breakdown
    version_counts = (
        db.query(TransactionRecord.model_version, func.count(TransactionRecord.id))
        .group_by(TransactionRecord.model_version)
        .all()
    )
    model_version_breakdown = {ver: int(count) for ver, count in version_counts}

    # Drift summary (uses cached drift evaluation to keep /stats fast)
    drift_data = get_cached_drift(db)
    top_feature = (
        drift_data["top_drifting_features"][0]["feature"]
        if drift_data.get("top_drifting_features")
        else None
    )
    top_psi = (
        drift_data["top_drifting_features"][0]["psi"]
        if drift_data.get("top_drifting_features")
        else None
    )

    drift_summary = DriftSummary(
        status=drift_data["status"],
        top_feature=top_feature,
        top_psi=top_psi,
    )

    return StatsResponse(
        total_scored=total_scored,
        flagged_count=flagged_count,
        flagged_rate=round(flagged_rate, 4),
        avg_latency_ms=round(avg_latency, 2),
        p95_latency_ms=round(p95_latency, 2),
        hourly_counts=sorted_hourly,
        drift_summary=drift_summary,
        model_version_breakdown=model_version_breakdown,
    )


# ── Drift Monitoring Cache & Logic ────────────────────────────────────
_drift_cache: dict[str, Any] | None = None
_drift_cache_time: float = 0.0
_last_logged_drift_status: str | None = None


def get_cached_drift(db: Session, force_refresh: bool = False) -> dict[str, Any]:
    """Retrieve or compute drift metrics with a 30-second TTL cache."""
    global _drift_cache, _drift_cache_time, _last_logged_drift_status
    now = time.time()

    from ml.config import DRIFT_CACHE_TTL_SECONDS
    from monitoring.drift import drift_monitor

    if (
        not force_refresh
        and _drift_cache is not None
        and (now - _drift_cache_time) < DRIFT_CACHE_TTL_SECONDS
    ):
        return _drift_cache

    try:
        drift_result = drift_monitor.evaluate_from_db(db)
    except Exception as e:
        logger.error(f"Error evaluating drift: {e}")
        return {
            "status": "insufficient_data",
            "window_size": 0,
            "min_rows_required": 200,
            "computed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "score_psi": 0.0,
            "flagged_rate": 0.0,
            "reference_flagged_rate": 0.0,
            "top_drifting_features": [],
        }

    _drift_cache = drift_result
    _drift_cache_time = now

    new_status = drift_result.get("status", "ok")
    if new_status in ("warning", "drift") and new_status != _last_logged_drift_status:
        top_feats = drift_result.get("top_drifting_features", [])
        top_desc = ", ".join(f"{f['feature']} (PSI={f['psi']})" for f in top_feats[:3])
        logger.warning(
            f"DRIFT ALERT: System status transitioned to '{new_status.upper()}'. "
            f"Top drifting features: {top_desc or 'None'}"
        )
        _last_logged_drift_status = new_status
    elif new_status == "ok" and _last_logged_drift_status in ("warning", "drift"):
        logger.info("Drift status recovered to 'OK'.")
        _last_logged_drift_status = "ok"

    return drift_result


@app.get("/drift", response_model=DriftResponse, tags=["Monitoring"])
def get_drift(db: Session = Depends(get_db)) -> DriftResponse:
    """Retrieve statistical data and score drift metrics over the rolling window."""
    result = get_cached_drift(db)
    return DriftResponse(**result)

