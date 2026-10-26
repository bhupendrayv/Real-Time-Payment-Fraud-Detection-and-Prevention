import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatCards } from '../components/StatCards';
import type { StatsResponse } from '../types';

describe('StatCards Component', () => {
  it('renders loading state when stats is null', () => {
    const { container } = render(<StatCards stats={null} loading={true} />);
    const pulses = container.querySelectorAll('.animate-pulse');
    expect(pulses.length).toBeGreaterThan(0);
  });

  it('renders metric cards with formatted numbers', () => {
    const mockStats: StatsResponse = {
      total_scored: 12450,
      flagged_count: 82,
      flagged_rate: 0.00658,
      avg_latency_ms: 14.28,
      p95_latency_ms: 22.50,
      hourly_counts: [],
    };

    render(<StatCards stats={mockStats} loading={false} />);

    expect(screen.getByText('12,450')).toBeInTheDocument();
    expect(screen.getByText('82')).toBeInTheDocument();
    expect(screen.getByText('0.66%')).toBeInTheDocument();
    expect(screen.getByText('14.3 ms')).toBeInTheDocument();
    expect(screen.getByText('22.5 ms')).toBeInTheDocument();
  });
});
