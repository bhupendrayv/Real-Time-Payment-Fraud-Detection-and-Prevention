import React, { useState } from 'react';
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
import type { PredictResponse, TransactionInput } from '../types';
import { predict } from '../api/client';
import { SAMPLE_FRAUD, SAMPLE_LEGITIMATE } from '../sampleData';

interface ManualTesterProps {
  isOpen: boolean;
  onClose: () => void;
  onPredictionSuccess?: (res: PredictResponse) => void;
}

export const ManualTester: React.FC<ManualTesterProps> = ({
  isOpen,
  onClose,
  onPredictionSuccess,
}) => {
  const [formData, setFormData] = useState<TransactionInput>(SAMPLE_LEGITIMATE);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  if (!isOpen) return null;

  const handleFieldChange = (key: keyof TransactionInput, val: string) => {
    const num = parseFloat(val);
    setFormData((prev) => ({
      ...prev,
      [key]: isNaN(num) ? 0 : num,
    }));
  };

  const handleLoadLegitimate = () => {
    setFormData(SAMPLE_LEGITIMATE);
    setResult(null);
    setError(null);
  };

  const handleLoadFraud = () => {
    setFormData(SAMPLE_FRAUD);
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await predict(formData, true);
      setResult(res);
      if (onPredictionSuccess) {
        onPredictionSuccess(res);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Prediction request failed');
    } finally {
      setLoading(false);
    }
  };

  const explanationData = (result?.explanation || []).map((exp) => ({
    feature: exp.feature,
    contribution: Number(exp.contribution.toFixed(4)),
  }));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-slate-950/80 backdrop-blur-sm"
        onClick={onClose}
      />

      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-10 overflow-hidden my-8 max-h-[92vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900/90">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              <span>Manual Transaction Simulator</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Score custom transactions through POST /predict with instant SHAP attribution
            </p>
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
        <div className="p-6 overflow-y-auto space-y-5 text-sm">
          {/* Quick presets */}
          <div className="flex flex-wrap items-center justify-between gap-3 bg-slate-950/60 p-3.5 rounded-xl border border-slate-800">
            <span className="text-xs text-slate-400 font-medium">
              Load realistic sample payload:
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleLoadLegitimate}
                className="px-3 py-1.5 rounded-lg bg-emerald-950/60 hover:bg-emerald-900/60 border border-emerald-800/60 text-emerald-300 text-xs font-semibold transition-colors flex items-center gap-1.5"
              >
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span>Legitimate Sample</span>
              </button>
              <button
                type="button"
                onClick={handleLoadFraud}
                className="px-3 py-1.5 rounded-lg bg-rose-950/60 hover:bg-rose-900/60 border border-rose-800/60 text-rose-300 text-xs font-semibold transition-colors flex items-center gap-1.5"
              >
                <span className="w-2 h-2 rounded-full bg-rose-400" />
                <span>Fraud Sample</span>
              </button>
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Amount ($)
                </label>
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  required
                  value={formData.Amount}
                  onChange={(e) => handleFieldChange('Amount', e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">
                  Time (seconds)
                </label>
                <input
                  type="number"
                  step="1"
                  min="0"
                  required
                  value={formData.Time}
                  onChange={(e) => handleFieldChange('Time', e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 font-mono focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
              </div>
            </div>

            {/* Collapsible V1 - V28 inputs */}
            <div className="border border-slate-800 rounded-xl overflow-hidden">
              <button
                type="button"
                onClick={() => setShowAdvanced(!showAdvanced)}
                className="w-full flex items-center justify-between px-4 py-2.5 bg-slate-800/40 text-xs font-medium text-slate-300 hover:bg-slate-800/60 transition-colors"
              >
                <span>PCA Feature Vectors (V1 - V28)</span>
                <span className="text-slate-500 text-[11px]">
                  {showAdvanced ? 'Collapse ▲' : 'Expand 28 features ▼'}
                </span>
              </button>

              {showAdvanced && (
                <div className="p-4 bg-slate-950/80 grid grid-cols-3 sm:grid-cols-4 gap-2.5 max-h-56 overflow-y-auto">
                  {Array.from({ length: 28 }, (_, i) => {
                    const key = `V${i + 1}` as keyof TransactionInput;
                    return (
                      <div key={key}>
                        <label className="block text-[10px] text-slate-400 font-mono">
                          {key}
                        </label>
                        <input
                          type="number"
                          step="0.001"
                          value={formData[key]}
                          onChange={(e) => handleFieldChange(key, e.target.value)}
                          className="w-full bg-slate-900 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 font-mono focus:outline-none focus:border-indigo-500"
                        />
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Error banner */}
            {error && (
              <div className="p-3 rounded-lg bg-rose-950/50 border border-rose-800/60 text-xs text-rose-300">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-all shadow-md shadow-indigo-600/30 flex items-center justify-center gap-2 disabled:opacity-50"
            >
              {loading ? (
                <>
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                  </svg>
                  <span>Scoring Transaction...</span>
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Score Transaction Now</span>
                </>
              )}
            </button>
          </form>

          {/* Inline Prediction Result */}
          {result && (
            <div className="space-y-4 pt-4 border-t border-slate-800">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Inference Decision Result
                </span>
                <span className="text-xs text-slate-400 font-mono">
                  Latency: <span className="text-cyan-400 font-bold">{result.latency_ms.toFixed(1)} ms</span>
                </span>
              </div>

              {/* Status pill card */}
              <div
                className={`p-4 rounded-xl border flex items-center justify-between ${
                  result.is_fraud
                    ? 'bg-rose-950/40 border-rose-800/70 text-rose-200'
                    : 'bg-emerald-950/40 border-emerald-800/70 text-emerald-200'
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`w-3 h-3 rounded-full ${
                        result.is_fraud ? 'bg-rose-500 animate-pulse' : 'bg-emerald-400'
                      }`}
                    />
                    <span className="text-base font-bold">
                      {result.is_fraud ? 'FLAGGED AS FRAUD' : 'TRANSACTION APPROVED (LEGITIMATE)'}
                    </span>
                  </div>
                  <span className="text-xs opacity-80 block mt-1">
                    Threshold: {(result.threshold * 100).toFixed(1)}% | Model: {result.model_version}
                  </span>
                </div>

                <div className="text-right">
                  <div className="text-2xl font-bold font-mono">
                    {(result.fraud_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-[10px] uppercase tracking-wider opacity-75">
                    Fraud Score
                  </div>
                </div>
              </div>

              {/* Top 5 SHAP attributions */}
              {explanationData.length > 0 && (
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800">
                  <div className="text-xs font-semibold text-slate-300 mb-2">
                    Top 5 Feature Attributions (SHAP)
                  </div>
                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        layout="vertical"
                        data={explanationData}
                        margin={{ top: 5, right: 30, left: 30, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} />
                        <YAxis type="category" dataKey="feature" stroke="#64748b" tick={{ fontSize: 11 }} />
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
                </div>
              )}
            </div>
          )}
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
