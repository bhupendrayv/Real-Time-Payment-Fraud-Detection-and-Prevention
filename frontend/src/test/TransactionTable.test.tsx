import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { TransactionTable } from '../components/TransactionTable';
import type { TransactionItem } from '../types';

const mockTransactions: TransactionItem[] = [
  {
    id: 'tx-1111-2222-3333-4444',
    created_at: '2026-10-24T12:00:00Z',
    features: { Amount: 25.5, Time: 100 },
    fraud_score: 0.05,
    is_fraud: false,
    threshold: 0.835,
    model_version: 'xgb_v1.0.0',
    explanation: [],
    latency_ms: 12.5,
  },
  {
    id: 'tx-5555-6666-7777-8888',
    created_at: '2026-10-24T12:05:00Z',
    features: { Amount: 499.99, Time: 120 },
    fraud_score: 0.94,
    is_fraud: true,
    threshold: 0.835,
    model_version: 'xgb_v1.0.0',
    explanation: [],
    latency_ms: 15.2,
  },
];

describe('TransactionTable Component', () => {
  it('renders transactions rows with legitimate and flagged badges', () => {
    render(
      <TransactionTable
        transactions={mockTransactions}
        loading={false}
        flaggedOnly={false}
        onToggleFlaggedOnly={vi.fn()}
        limit={50}
        onChangeLimit={vi.fn()}
        onSelectTransaction={vi.fn()}
      />
    );

    expect(screen.getByText(/tx-1111/)).toBeInTheDocument();
    expect(screen.getByText(/tx-5555/)).toBeInTheDocument();
    expect(screen.getByText('$25.50')).toBeInTheDocument();
    expect(screen.getByText('$499.99')).toBeInTheDocument();
    expect(screen.getByText('Legitimate')).toBeInTheDocument();
    expect(screen.getByText('Flagged')).toBeInTheDocument();
  });

  it('triggers onToggleFlaggedOnly when clicking filter buttons', () => {
    const handleToggle = vi.fn();
    render(
      <TransactionTable
        transactions={mockTransactions}
        loading={false}
        flaggedOnly={false}
        onToggleFlaggedOnly={handleToggle}
        limit={50}
        onChangeLimit={vi.fn()}
        onSelectTransaction={vi.fn()}
      />
    );

    const flaggedOnlyBtn = screen.getByRole('button', { name: /Flagged Only/i });
    fireEvent.click(flaggedOnlyBtn);
    expect(handleToggle).toHaveBeenCalledWith(true);
  });

  it('triggers onSelectTransaction when row is clicked', () => {
    const handleSelect = vi.fn();
    render(
      <TransactionTable
        transactions={mockTransactions}
        loading={false}
        flaggedOnly={false}
        onToggleFlaggedOnly={vi.fn()}
        limit={50}
        onChangeLimit={vi.fn()}
        onSelectTransaction={handleSelect}
      />
    );

    const row = screen.getByText(/tx-5555/).closest('tr');
    expect(row).not.toBeNull();
    if (row) {
      fireEvent.click(row);
      expect(handleSelect).toHaveBeenCalledWith(mockTransactions[1]);
    }
  });
});
