import React, { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import type { TransactionItem } from '../types';

interface TransactionModalProps {
  transaction: TransactionItem | null;
  onClose: () => void;
}

export const TransactionModal: React.FC<TransactionModalProps> = ({
  transaction,
  onClose,
}) => {
  const [showRawFeatures, setShowRawFeatures] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  if (!transaction) return null;

  const copyId = () => {
    navigator.clipboard.writeText(transaction.id);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Prepare SHAP chart data
  const explanationData = (transaction.explanation || []).map((exp) => ({
    feature: exp.feature,
    contribution: Number(exp.contribution.toFixed(4)),
  }));

  const amount =
    typeof transaction.features?.Amount === 'number'
      ? transaction.features.Amount
      : 0;

  const timeVal =
    typeof transaction.features?.Time === 'number'
      ? transaction.features.Time
      : 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm transition-opacity"
        onClick={onClose}
      />

      {/* Modal Dialog */}
      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-10 overflow-hidden my-8 max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/90">
          <div className="flex items-center gap-3">
            <span
              className={`px-2.5 py-1 rounded-full text-xs font-semibold ${
                transaction.is_fraud
                  ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                  : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
              }`}
            >
              {transaction.is_fraud ? 'FLAGGED FRAUD' : 'LEGITIMATE'}
            </span>
            <h3 className="text-base font-semibold text-white">
              Transaction Details
            </h3>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content body */}
        <div className="p-6 overflow-y-auto space-y-6 text-sm">
          {/* Top metadata grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/60">
              <span className="text-[11px] text-slate-400 block">Fraud Score</span>
              <span
                className={`text-lg font-bold font-mono ${
                  transaction.is_fraud ? 'text-rose-400' : 'text-emerald-400'
                }`}
              >
                {(transaction.fraud_score * 100).toFixed(1)}%
              </span>
            </div>

            <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/60">
              <span className="text-[11px] text-slate-400 block">Amount</span>
              <span className="text-lg font-bold font-mono text-white">
                ${amount.toFixed(2)}
              </span>
            </div>

            <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/60">
              <span className="text-[11px] text-slate-400 block">Threshold</span>
              <span className="text-lg font-bold font-mono text-indigo-300">
                {(transaction.threshold * 100).toFixed(1)}%
              </span>
            </div>

            <div className="bg-slate-800/50 p-3 rounded-xl border border-slate-700/60">
              <span className="text-[11px] text-slate-400 block">Inference Latency</span>
              <span className="text-lg font-bold font-mono text-cyan-400">
                {transaction.latency_ms.toFixed(1)} ms
              </span>
            </div>
          </div>

          {/* ID & Time info */}
          <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800 space-y-1.5 text-xs font-mono">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Transaction ID:</span>
              <div className="flex items-center gap-2">
                <span className="text-slate-200 select-all">{transaction.id}</span>
                <button
                  type="button"
                  onClick={copyId}
                  className="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] transition-colors"
                >
                  {copied ? 'Copied!' : 'Copy'}
                </button>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Created At:</span>
              <span className="text-slate-300">
                {new Date(transaction.created_at).toLocaleString()}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Time (Dataset relative):</span>
              <span className="text-slate-300">{timeVal.toFixed(1)}s</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Model Version:</span>
              <span className="text-slate-300">{transaction.model_version}</span>
            </div>
          </div>

          {/* SHAP Explanation Section */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">
                  SHAP Feature Attributions (Top 5)
                </h4>
                <p className="text-xs text-slate-400">
                  Explains why the model scored this transaction as{' '}
                  {transaction.is_fraud ? 'fraudulent' : 'legitimate'}.
                </p>
              </div>
            </div>

            {explanationData.length > 0 ? (
              <div className="bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <div className="h-48 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={explanationData}
                      margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis
                        type="number"
                        stroke="#64748b"
                        tick={{ fontSize: 10 }}
                        domain={['auto', 'auto']}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        stroke="#64748b"
                        tick={{ fontSize: 11, fill: '#cbd5e1' }}
                        width={65}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0f172a',
                          borderColor: '#334155',
                          borderRadius: '8px',
                          fontSize: '11px',
                          color: '#f8fafc',
                        }}
                        formatter={(val: unknown) => [
                          `${Number(val) > 0 ? '+' : ''}${String(val)} (${Number(val) > 0 ? 'Pushes toward Fraud' : 'Pushes toward Legitimate'})`,
                          'SHAP Value',
                        ]}
                      />
                      <ReferenceLine x={0} stroke="#475569" />
                      <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                        {explanationData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.contribution > 0 ? '#ef4444' : '#10b981'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <div className="flex items-center justify-center gap-6 mt-2 text-[11px] text-slate-400">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" />
                    <span>Positive SHAP: Increases fraud risk</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" />
                    <span>Negative SHAP: Decreases fraud risk</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-950/40 text-xs text-slate-400 text-center">
                No feature explanations were generated for this transaction.
              </div>
            )}
          </div>

          {/* Raw features toggle */}
          <div>
            <button
              type="button"
              onClick={() => setShowRawFeatures(!showRawFeatures)}
              className="flex items-center gap-2 text-xs font-semibold text-indigo-400 hover:text-indigo-300 transition-colors"
            >
              <svg
                className={`w-4 h-4 transform transition-transform ${
                  showRawFeatures ? 'rotate-90' : ''
                }`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7" />
              </svg>
              <span>{showRawFeatures ? 'Hide Raw Feature Vector' : 'View Full Feature Vector (30 inputs)'}</span>
            </button>

            {showRawFeatures && (
              <div className="mt-3 p-3 bg-slate-950 rounded-xl border border-slate-800 max-h-48 overflow-y-auto">
                <pre className="text-[11px] font-mono text-slate-300 leading-relaxed">
                  {JSON.stringify(transaction.features, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-900/90 flex justify-end">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
