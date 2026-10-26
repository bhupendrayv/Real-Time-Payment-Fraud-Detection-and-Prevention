import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DriftCard } from '../components/DriftCard';
import type { DriftResponse } from '../types';

describe('DriftCard Component', () => {
  it('renders insufficient data badge when fewer than 200 records exist', () => {
    const mockDrift: DriftResponse = {
      status: 'insufficient_data',
      window_size: 45,
      min_rows_required: 200,
      computed_at: '2026-10-24T12:00:00Z',
      score_psi: 0.0,
      flagged_rate: 0.0,
      reference_flagged_rate: 0.0016,
      top_drifting_features: [],
    };

    render(<DriftCard drift={mockDrift} loading={false} />);
    expect(screen.getByText(/NOT ENOUGH DATA/i)).toBeInTheDocument();
    expect(screen.getByText(/Accumulating transactions/i)).toBeInTheDocument();
  });

  it('renders STABLE (OK) badge when distribution matches baseline', () => {
    const mockDrift: DriftResponse = {
      status: 'ok',
      window_size: 500,
      min_rows_required: 200,
      computed_at: '2026-10-24T12:00:00Z',
      score_psi: 0.0215,
      flagged_rate: 0.0018,
      reference_flagged_rate: 0.0016,
      top_drifting_features: [
        {
          feature: 'V14',
          psi: 0.045,
          ks_statistic: 0.03,
          ks_pvalue: 0.42,
          status: 'ok',
        },
      ],
    };

    render(<DriftCard drift={mockDrift} loading={false} />);
    expect(screen.getByText(/STABLE \(OK\)/i)).toBeInTheDocument();
    expect(screen.getByText('0.0215')).toBeInTheDocument();
    expect(screen.getByText('V14')).toBeInTheDocument();
  });

  it('renders DRIFT DETECTED badge and displays drifting features', () => {
    const mockDrift: DriftResponse = {
      status: 'drift',
      window_size: 500,
      min_rows_required: 200,
      computed_at: '2026-10-24T12:00:00Z',
      score_psi: 0.3842,
      flagged_rate: 0.045,
      reference_flagged_rate: 0.0016,
      top_drifting_features: [
        {
          feature: 'V14',
          psi: 0.4821,
          ks_statistic: 0.35,
          ks_pvalue: 0.00001,
          status: 'drift',
        },
        {
          feature: 'Amount',
          psi: 0.125,
          ks_statistic: 0.08,
          ks_pvalue: 0.04,
          status: 'warning',
        },
      ],
    };

    render(<DriftCard drift={mockDrift} loading={false} />);
    expect(screen.getByText(/DRIFT DETECTED/i)).toBeInTheDocument();
    expect(screen.getByText('0.3842')).toBeInTheDocument();
    expect(screen.getByText('V14')).toBeInTheDocument();
    expect(screen.getByText('0.4821')).toBeInTheDocument();
  });
});
