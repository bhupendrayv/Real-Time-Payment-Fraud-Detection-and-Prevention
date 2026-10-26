import React, { useState } from 'react';
import type { TransactionItem } from '../types';

interface TransactionTableProps {
  transactions: TransactionItem[];
  loading: boolean;
  flaggedOnly: boolean;
  onToggleFlaggedOnly: (flaggedOnly: boolean) => void;
  limit: number;
  onChangeLimit: (limit: number) => void;
  onSelectTransaction: (tx: TransactionItem) => void;
}

export const TransactionTable: React.FC<TransactionTableProps> = ({
  transactions,
  loading,
  flaggedOnly,
  onToggleFlaggedOnly,
  limit,
  onChangeLimit,
  onSelectTransaction,
}) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleCopy = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 overflow-hidden backdrop-blur-sm">
      {/* Table controls header */}
      <div className="p-4 sm:p-5 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <span>Recent Transactions Stream</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
              {transactions.length} records
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Click any row to inspect SHAP attributions and feature details
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Filter toggle */}
          <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700/60 text-xs">
            <button
              type="button"
              onClick={() => onToggleFlaggedOnly(false)}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                !flaggedOnly
                  ? 'bg-indigo-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              All Transactions
            </button>
            <button
              type="button"
              onClick={() => onToggleFlaggedOnly(true)}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                flaggedOnly
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-rose-300'
              }`}
            >
              Flagged Only
            </button>
          </div>

          {/* Row limit dropdown */}
          <select
            value={limit}
            onChange={(e) => onChangeLimit(Number(e.target.value))}
            className="bg-slate-800 border border-slate-700/80 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-1 focus:ring-indigo-500"
          >
            <option value={25}>25 rows</option>
            <option value={50}>50 rows</option>
            <option value={100}>100 rows</option>
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="bg-slate-950/50 text-slate-400 font-semibold border-b border-slate-800/80 uppercase tracking-wider text-[11px]">
            <tr>
              <th className="px-4 py-3">Timestamp</th>
              <th className="px-4 py-3">Transaction ID</th>
              <th className="px-4 py-3 text-right">Amount</th>
              <th className="px-4 py-3 text-center">Fraud Probability</th>
              <th className="px-4 py-3 text-center">Status</th>
              <th className="px-4 py-3 text-right">Latency</th>
              <th className="px-4 py-3 text-center">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50 font-mono">
            {loading && transactions.length === 0 ? (
              [...Array(5)].map((_, i) => (
                <tr key={i} className="animate-pulse">
                  <td colSpan={7} className="px-4 py-4">
                    <div className="h-4 bg-slate-800/60 rounded w-full" />
                  </td>
                </tr>
              ))
            ) : transactions.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center text-slate-400 font-sans">
                  <div className="flex flex-col items-center justify-center gap-2">
                    <svg
                      className="w-8 h-8 text-slate-500"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="1.5"
                        d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"
                      />
                    </svg>
                    <span>No transactions found matching the current filter.</span>
                    <span className="text-xs text-slate-500">
                      Run the simulator or test a manual transaction to see live incoming data.
                    </span>
                  </div>
                </td>
              </tr>
            ) : (
              transactions.map((tx) => {
                const amount =
                  typeof tx.features?.Amount === 'number'
                    ? tx.features.Amount
                    : 0;

                const scorePct = (tx.fraud_score * 100).toFixed(1);

                return (
                  <tr
                    key={tx.id}
                    onClick={() => onSelectTransaction(tx)}
                    className="hover:bg-slate-800/40 cursor-pointer transition-colors group"
                  >
                    {/* Timestamp */}
                    <td className="px-4 py-3 text-slate-300 font-sans text-xs whitespace-nowrap">
                      {new Date(tx.created_at).toLocaleTimeString()}
                    </td>

                    {/* ID with copy */}
                    <td className="px-4 py-3 text-slate-300 whitespace-nowrap">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-indigo-300 group-hover:underline">
                          {tx.id.substring(0, 8)}...{tx.id.substring(tx.id.length - 4)}
                        </span>
                        <button
                          type="button"
                          onClick={(e) => handleCopy(e, tx.id)}
                          className="opacity-0 group-hover:opacity-100 p-1 hover:bg-slate-700 rounded text-slate-400 hover:text-white transition-opacity"
                          title="Copy full transaction ID"
                        >
                          {copiedId === tx.id ? (
                            <span className="text-[10px] text-emerald-400 font-sans">Copied!</span>
                          ) : (
                            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth="2"
                                d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                              />
                            </svg>
                          )}
                        </button>
                      </div>
                    </td>

                    {/* Amount */}
                    <td className="px-4 py-3 text-right text-slate-100 font-semibold whitespace-nowrap">
                      ${amount.toFixed(2)}
                    </td>

                    {/* Fraud score progress */}
                    <td className="px-4 py-3 whitespace-nowrap">
                      <div className="flex items-center justify-center gap-2">
                        <div className="w-16 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              tx.is_fraud ? 'bg-rose-500' : 'bg-emerald-500'
                            }`}
                            style={{ width: `${Math.min(100, tx.fraud_score * 100)}%` }}
                          />
                        </div>
                        <span
                          className={`text-xs font-semibold ${
                            tx.is_fraud ? 'text-rose-400' : 'text-slate-300'
                          }`}
                        >
                          {scorePct}%
                        </span>
                      </div>
                    </td>

                    {/* Badge */}
                    <td className="px-4 py-3 text-center whitespace-nowrap font-sans">
                      <span
                        className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold border ${
                          tx.is_fraud
                            ? 'bg-rose-500/15 border-rose-500/30 text-rose-300'
                            : 'bg-emerald-500/15 border-emerald-500/30 text-emerald-300'
                        }`}
                      >
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            tx.is_fraud ? 'bg-rose-400' : 'bg-emerald-400'
                          }`}
                        />
                        {tx.is_fraud ? 'Flagged' : 'Legitimate'}
                      </span>
                    </td>

                    {/* Latency */}
                    <td className="px-4 py-3 text-right text-slate-400 whitespace-nowrap">
                      {tx.latency_ms.toFixed(1)} ms
                    </td>

                    {/* Action */}
                    <td className="px-4 py-3 text-center whitespace-nowrap font-sans">
                      <span className="text-xs text-indigo-400 group-hover:text-indigo-300 font-medium">
                        Inspect &rarr;
                      </span>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
};
