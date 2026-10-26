/**
 * TypeScript types matching FastAPI Pydantic schemas exactly.
 */

export interface TransactionInput {
  Time: number;
  Amount: number;
  V1: number;
  V2: number;
  V3: number;
  V4: number;
  V5: number;
  V6: number;
  V7: number;
  V8: number;
  V9: number;
  V10: number;
  V11: number;
  V12: number;
  V13: number;
  V14: number;
  V15: number;
  V16: number;
  V17: number;
  V18: number;
  V19: number;
  V20: number;
  V21: number;
  V22: number;
  V23: number;
  V24: number;
  V25: number;
  V26: number;
  V27: number;
  V28: number;
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
}

export interface PredictResponse {
  transaction_id: string;
  fraud_score: number;
  is_fraud: boolean;
  threshold: number;
  model_version: string;
  explanation: FeatureContribution[];
  latency_ms: number;
}

export interface HealthResponse {
  status: string;
  model_version: string;
}

export interface TransactionItem {
  id: string;
  created_at: string;
  features: Record<string, number>;
  fraud_score: number;
  is_fraud: boolean;
  threshold: number;
  model_version: string;
  explanation: FeatureContribution[] | null;
  latency_ms: number;
}

export interface HourlyCount {
  hour: string;
  total: number;
  flagged: number;
}

export interface StatsResponse {
  total_scored: number;
  flagged_count: number;
  flagged_rate: number;
  avg_latency_ms: number;
  p95_latency_ms: number;
  hourly_counts: HourlyCount[];
}

export interface FeatureDriftDetail {
  feature: string;
  psi: number;
  ks_statistic: number;
  ks_pvalue: number;
  status: string;
}

export interface DriftResponse {
  status: 'ok' | 'warning' | 'drift' | 'insufficient_data';
  window_size: number;
  min_rows_required: number;
  computed_at: string;
  score_psi: number;
  flagged_rate: number;
  reference_flagged_rate: number;
  top_drifting_features: FeatureDriftDetail[];
}

