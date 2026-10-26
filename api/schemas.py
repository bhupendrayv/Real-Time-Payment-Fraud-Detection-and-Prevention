"""Pydantic validation schemas for FraudShield API."""
from __future__ import annotations

import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class TransactionInput(BaseModel):
    """Transaction features required for real-time fraud inference."""

    Time: float = Field(..., description="Seconds elapsed since the first transaction in the dataset")
    Amount: float = Field(..., ge=0.0, description="Transaction amount in currency units (must be >= 0)")

    # PCA anonymized features V1 - V28
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Time": 406.0,
                "Amount": 149.62,
                "V1": -2.31,
                "V2": 1.95,
                "V3": -1.61,
                "V4": 3.99,
                "V5": -0.52,
                "V6": -1.43,
                "V7": -2.54,
                "V8": 1.39,
                "V9": -2.77,
                "V10": -2.77,
                "V11": 3.20,
                "V12": -2.90,
                "V13": -0.60,
                "V14": -4.29,
                "V15": 0.39,
                "V16": -1.14,
                "V17": -2.83,
                "V18": -0.02,
                "V19": 0.42,
                "V20": 0.13,
                "V21": 0.52,
                "V22": -0.04,
                "V23": -0.47,
                "V24": 0.32,
                "V25": 0.04,
                "V26": 0.18,
                "V27": 0.26,
                "V28": -0.14,
            }
        },
    )


class FeatureContribution(BaseModel):
    """Feature attribution returned by TreeExplainer."""

    feature: str = Field(..., description="Feature name")
    contribution: float = Field(
        ...,
        description="SHAP attribution value (positive increases fraud probability, negative decreases)",
    )


class PredictResponse(BaseModel):
    """Inference result returned to client."""

    transaction_id: str = Field(..., description="Unique UUID assigned to this transaction")
    fraud_score: float = Field(..., ge=0.0, le=1.0, description="Estimated fraud probability")
    is_fraud: bool = Field(..., description="True if fraud_score >= decision threshold")
    threshold: float = Field(..., description="Operational decision threshold chosen to minimize expected cost")
    model_version: str = Field(..., description="Model identifier/version used for inference")
    explanation: list[FeatureContribution] = Field(
        default_factory=list,
        description="Top 5 feature contributions explaining the prediction (empty if explain=false)",
    )
    latency_ms: float = Field(..., description="Processing latency inside /predict in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    model_version: str


class TransactionItem(BaseModel):
    """Transaction record summary for audit logs."""

    id: str
    created_at: datetime.datetime
    features: dict[str, Any]
    fraud_score: float
    is_fraud: bool
    threshold: float
    model_version: str
    explanation: list[FeatureContribution] | None = None
    latency_ms: float

    model_config = ConfigDict(from_attributes=True)


class HourlyCount(BaseModel):
    """Hourly aggregation data point."""

    hour: str = Field(..., description="Hour bucket formatted as YYYY-MM-DD HH:00")
    total: int = Field(..., description="Total transactions scored in this hour")
    flagged: int = Field(..., description="Total transactions flagged as fraudulent in this hour")


class FeatureDriftDetail(BaseModel):
    """Statistical drift metrics for an individual feature."""

    feature: str = Field(..., description="Feature name")
    psi: float = Field(..., description="Population Stability Index")
    ks_statistic: float = Field(..., description="Kolmogorov-Smirnov test statistic")
    ks_pvalue: float = Field(..., description="Kolmogorov-Smirnov p-value")
    status: str = Field(..., description="Alert status: ok, warning, or drift")


class DriftResponse(BaseModel):
    """Drift evaluation response for GET /drift."""

    status: str = Field(..., description="Overall drift status: ok, warning, drift, or insufficient_data")
    window_size: int = Field(..., description="Number of recent transactions analyzed")
    min_rows_required: int = Field(default=200, description="Minimum rows needed to evaluate drift")
    computed_at: str = Field(..., description="ISO 8601 timestamp of drift calculation")
    score_psi: float = Field(..., description="PSI on fraud score distribution")
    flagged_rate: float = Field(..., description="Observed flagged rate in recent window")
    reference_flagged_rate: float = Field(..., description="Baseline reference flagged rate from validation split")
    top_drifting_features: list[FeatureDriftDetail] = Field(
        default_factory=list,
        description="Top 5 drifting features sorted by PSI descending",
    )


class DriftSummary(BaseModel):
    """Brief drift overview included in GET /stats."""

    status: str = Field(..., description="Overall drift status")
    top_feature: str | None = Field(default=None, description="Name of the highest-PSI feature")
    top_psi: float | None = Field(default=None, description="PSI of highest-PSI feature")


class StatsResponse(BaseModel):
    """Aggregated operational metrics for dashboard."""

    total_scored: int = Field(..., description="Total number of transactions processed")
    flagged_count: int = Field(..., description="Total number of transactions flagged as fraud")
    flagged_rate: float = Field(..., description="Ratio of flagged transactions to total scored")
    avg_latency_ms: float = Field(..., description="Average inference latency in milliseconds")
    p95_latency_ms: float = Field(..., description="95th percentile inference latency in milliseconds")
    hourly_counts: list[HourlyCount] = Field(
        default_factory=list,
        description="Transaction counts grouped by hour for the past 24 hours",
    )
    drift_summary: DriftSummary | None = Field(
        default=None,
        description="Current drift status summary",
    )
    model_version_breakdown: dict[str, int] = Field(
        default_factory=dict,
        description="Count of processed transactions per model version",
    )

