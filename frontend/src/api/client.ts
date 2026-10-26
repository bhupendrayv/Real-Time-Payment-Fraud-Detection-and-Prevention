import type {
  DriftResponse,
  HealthResponse,
  PredictResponse,
  StatsResponse,
  TransactionInput,
  TransactionItem,
} from '../types';

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, '') ||
  'http://localhost:8000';

async function handleResponse<T>(res: Response, endpoint: string): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const data = await res.json();
      if (data && typeof data === 'object' && 'detail' in data) {
        errorDetail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch {
      // Body wasn't JSON, retain statusText
    }
    throw new Error(`Failed to fetch ${endpoint} - ${errorDetail}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/health`, {
    headers: { Accept: 'application/json' },
  });
  return handleResponse<HealthResponse>(res, '/health');
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${API_BASE_URL}/stats`, {
    headers: { Accept: 'application/json' },
  });
  return handleResponse<StatsResponse>(res, '/stats');
}

export async function getTransactions(
  limit: number = 50,
  flaggedOnly: boolean = false
): Promise<TransactionItem[]> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    flagged_only: flaggedOnly ? 'true' : 'false',
  });
  const res = await fetch(`${API_BASE_URL}/transactions?${params.toString()}`, {
    headers: { Accept: 'application/json' },
  });
  return handleResponse<TransactionItem[]>(res, '/transactions');
}

export async function predict(
  payload: TransactionInput,
  explain: boolean = true
): Promise<PredictResponse> {
  const params = new URLSearchParams({
    explain: explain ? 'true' : 'false',
  });
  const res = await fetch(`${API_BASE_URL}/predict?${params.toString()}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    body: JSON.stringify(payload),
  });
  return handleResponse<PredictResponse>(res, '/predict');
}

export async function getDrift(): Promise<DriftResponse> {
  const res = await fetch(`${API_BASE_URL}/drift`, {
    headers: { Accept: 'application/json' },
  });
  return handleResponse<DriftResponse>(res, '/drift');
}

export interface ImagePredictResponse extends PredictResponse {
  ocr_available: boolean;
  ocr_text: string | null;
  extracted_fields: {
    amount: number | null;
    time_seconds: number | null;
    card_last4: string | null;
    merchant: string | null;
  };
  fields_used: { Amount: number; Time: number };
}

export async function predictFromImage(
  file: File,
  explain: boolean = true
): Promise<ImagePredictResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const params = new URLSearchParams({ explain: explain ? 'true' : 'false' });
  const res = await fetch(`${API_BASE_URL}/predict/from-image?${params}`, {
    method: 'POST',
    headers: { Accept: 'application/json' },
    body: formData,
  });
  return handleResponse<ImagePredictResponse>(res, '/predict/from-image');
}


