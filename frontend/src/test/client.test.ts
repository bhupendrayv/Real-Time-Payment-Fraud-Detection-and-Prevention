import { describe, it, expect, vi, beforeEach } from 'vitest';
import { getHealth, getStats, getTransactions, predict, API_BASE_URL } from '../api/client';
import { SAMPLE_LEGITIMATE } from '../sampleData';

describe('API Client', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('calls /health and returns health status', async () => {
    const mockHealth = { status: 'ok', model_version: 'xgb_v1.0.0' };
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const res = await getHealth();
    expect(res).toEqual(mockHealth);
    expect(globalThis.fetch).toHaveBeenCalledWith(`${API_BASE_URL}/health`, expect.any(Object));
  });

  it('calls /stats and returns aggregated stats', async () => {
    const mockStats = {
      total_scored: 100,
      flagged_count: 5,
      flagged_rate: 0.05,
      avg_latency_ms: 12.4,
      p95_latency_ms: 18.2,
      hourly_counts: [],
    };
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockStats,
    } as Response);

    const res = await getStats();
    expect(res).toEqual(mockStats);
    expect(globalThis.fetch).toHaveBeenCalledWith(`${API_BASE_URL}/stats`, expect.any(Object));
  });

  it('calls /transactions with correct query parameters', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => [],
    } as Response);

    await getTransactions(25, true);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      `${API_BASE_URL}/transactions?limit=25&flagged_only=true`,
      expect.any(Object)
    );
  });

  it('calls /predict with JSON payload and returns scored response', async () => {
    const mockPredictRes = {
      transaction_id: 'test-uuid-1234',
      fraud_score: 0.02,
      is_fraud: false,
      threshold: 0.835,
      model_version: 'xgb_v1.0.0',
      explanation: [{ feature: 'V14', contribution: -0.4 }],
      latency_ms: 8.5,
    };

    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: true,
      json: async () => mockPredictRes,
    } as Response);

    const res = await predict(SAMPLE_LEGITIMATE, true);
    expect(res).toEqual(mockPredictRes);
    expect(globalThis.fetch).toHaveBeenCalledWith(
      `${API_BASE_URL}/predict?explain=true`,
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
      })
    );
  });

  it('throws descriptive error on HTTP failure', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ detail: 'Model failed to compute inference' }),
    } as Response);

    await expect(getHealth()).rejects.toThrow('Failed to fetch /health');
  });
});
