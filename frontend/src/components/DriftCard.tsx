import React from 'react';
import type { DriftResponse } from '../types';

interface DriftCardProps {
  drift: DriftResponse | null;
  loading: boolean;
  error?: string | null;
}

export const DriftCard: React.FC<DriftCardProps> = ({ drift, loading, error }) => {
  if (loading && !drift) {
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm animate-pulse">
        <div className="flex items-center justify-between mb-4">
          <div className="h-5 w-40 bg-slate-800 rounded" />
          <div className="h-6 w-24 bg-slate-800 rounded-full" />
        </div>
        <div className="grid grid-cols-2 gap-3 mb-4">
          <div className="h-16 bg-slate-800/60 rounded-lg" />
          <div className="h-16 bg-slate-800/60 rounded-lg" />
        </div>
        <div className="h-20 bg-slate-800/40 rounded-lg" />
      </div>
    );
  }

  if (error && !drift) {
    return (
      <div className="rounded-xl border border-rose-900/50 bg-rose-950/20 p-5 text-xs text-rose-300">
        <div className="font-semibold text-rose-200 mb-1">Drift Monitor Offline</div>
        <div>Unable to fetch drift statistics from API: {error}</div>
      </div>
    );
  }

  const status = drift?.status ?? 'insufficient_data';
  const scorePsi = drift?.score_psi ?? 0.0;
  const flaggedRate = drift ? (drift.flagged_rate * 100).toFixed(2) : '0.00';
  const refFlaggedRate = drift ? (drift.reference_flagged_rate * 100).toFixed(2) : '0.16';
  const topFeatures = drift?.top_drifting_features?.slice(0, 3) ?? [];

  const getBadge = () => {
    switch (status) {
      case 'ok':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 border border-emerald-500/30 text-emerald-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            STABLE (OK)
          </span>
        );
      case 'warning':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/15 border border-amber-500/30 text-amber-300">
            <span className="w-2 h-2 rounded-full bg-amber-400" />
            WARNING (0.10 ≤ PSI ≤ 0.25)
          </span>
        );
      case 'drift':
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/15 border border-rose-500/30 text-rose-300 animate-pulse">
            <span className="w-2 h-2 rounded-full bg-rose-400" />
            DRIFT DETECTED (PSI &gt; 0.25)
          </span>
        );
      case 'insufficient_data':
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-800 border border-slate-700 text-slate-400">
            <span className="w-2 h-2 rounded-full bg-slate-500" />
            NOT ENOUGH DATA (&lt; 200 TXS)
          </span>
        );
    }
  };

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-4">
        <div>
          <h3 className="text-base font-semibold text-white flex items-center gap-2">
            <svg
              className="w-5 h-5 text-cyan-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            <span>Population &amp; Model Drift Monitor</span>
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Statistical PSI and KS tests on rolling window (Window: {drift?.window_size ?? 0} txs)
          </p>
        </div>

        <div>{getBadge()}</div>
      </div>

      {/* Primary KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        {/* Score PSI */}
        <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/80 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block">Fraud Score PSI</span>
            <span className="text-xs text-slate-500 font-mono">Baseline vs Live Window</span>
          </div>
          <div className="text-right">
            <span
              className={`text-xl font-bold font-mono ${
                scorePsi > 0.25
                  ? 'text-rose-400'
                  : scorePsi >= 0.1
                  ? 'text-amber-400'
                  : 'text-emerald-400'
              }`}
            >
              {scorePsi.toFixed(4)}
            </span>
            <span className="text-[10px] text-slate-500 block">Threshold: 0.10</span>
          </div>
        </div>

        {/* Flagged Rate Comparison */}
        <div className="bg-slate-950/50 p-3 rounded-xl border border-slate-800/80 flex items-center justify-between">
          <div>
            <span className="text-[11px] text-slate-400 block">Flagged Rate (Live vs Ref)</span>
            <span className="text-xs text-slate-500 font-mono">Reference: {refFlaggedRate}%</span>
          </div>
          <div className="text-right">
            <span className="text-xl font-bold font-mono text-white">{flaggedRate}%</span>
            <span className="text-[10px] text-slate-500 block">Current Window</span>
          </div>
        </div>
      </div>

      {/* Top 3 Drifting Features List */}
      <div>
        <div className="text-xs font-semibold text-slate-300 mb-2">
          Top Drifting Features (PSI Rank)
        </div>

        {status === 'insufficient_data' ? (
          <div className="p-3 rounded-lg bg-slate-950/30 border border-slate-800/50 text-xs text-slate-400 text-center font-mono">
            Accumulating transactions... Need at least 200 records to compute baseline drift ({drift?.window_size ?? 0}/200).
          </div>
        ) : topFeatures.length === 0 ? (
          <div className="p-3 rounded-lg bg-slate-950/30 border border-slate-800/50 text-xs text-slate-400 text-center">
            All feature distributions match the training baseline. No drift detected.
          </div>
        ) : (
          <div className="space-y-2">
            {topFeatures.map((f, idx) => (
              <div
                key={f.feature}
                className="flex items-center justify-between px-3 py-2 rounded-lg bg-slate-950/60 border border-slate-800/80 text-xs font-mono"
              >
                <div className="flex items-center gap-2">
                  <span className="text-slate-500 text-[11px]">#{idx + 1}</span>
                  <span className="font-semibold text-indigo-300">{f.feature}</span>
                </div>

                <div className="flex items-center gap-4">
                  <div className="text-slate-400">
                    PSI: <span className="text-white font-bold">{f.psi.toFixed(4)}</span>
                  </div>
                  <div className="text-slate-400 hidden sm:inline">
                    KS p-val:{' '}
                    <span className="text-slate-300">{f.ks_pvalue.toFixed(4)}</span>
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-sans font-semibold uppercase ${
                      f.status === 'drift'
                        ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                        : f.status === 'warning'
                        ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                        : 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                    }`}
                  >
                    {f.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
