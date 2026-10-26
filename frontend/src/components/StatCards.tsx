import React from 'react';
import type { StatsResponse } from '../types';

interface StatCardsProps {
  stats: StatsResponse | null;
  loading: boolean;
}

export const StatCards: React.FC<StatCardsProps> = ({ stats, loading }) => {
  if (loading && !stats) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className="h-28 rounded-xl border border-slate-800 bg-slate-900/50 p-4 animate-pulse flex flex-col justify-between"
          >
            <div className="h-4 w-20 bg-slate-800 rounded" />
            <div className="h-8 w-28 bg-slate-800 rounded" />
            <div className="h-3 w-16 bg-slate-800 rounded" />
          </div>
        ))}
      </div>
    );
  }

  const totalScored = stats?.total_scored ?? 0;
  const flaggedCount = stats?.flagged_count ?? 0;
  const flaggedRatePct = stats ? (stats.flagged_rate * 100).toFixed(2) : '0.00';
  const avgLatency = stats ? stats.avg_latency_ms.toFixed(1) : '0.0';
  const p95Latency = stats ? stats.p95_latency_ms.toFixed(1) : '0.0';

  const cards = [
    {
      label: 'Total Scored',
      value: totalScored.toLocaleString(),
      subtext: 'Processed transactions',
      color: 'from-blue-500/20 to-indigo-500/10 text-blue-400 border-blue-500/30',
      icon: (
        <svg className="w-5 h-5 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      label: 'Flagged Fraud',
      value: flaggedCount.toLocaleString(),
      subtext: 'Flagged for review',
      color: 'from-rose-500/20 to-red-500/10 text-rose-400 border-rose-500/30',
      icon: (
        <svg className="w-5 h-5 text-rose-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
      ),
    },
    {
      label: 'Flagged %',
      value: `${flaggedRatePct}%`,
      subtext: 'Detection rate',
      color: 'from-amber-500/20 to-yellow-500/10 text-amber-400 border-amber-500/30',
      icon: (
        <svg className="w-5 h-5 text-amber-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" />
        </svg>
      ),
    },
    {
      label: 'Avg Latency',
      value: `${avgLatency} ms`,
      subtext: 'Mean processing time',
      color: 'from-emerald-500/20 to-teal-500/10 text-emerald-400 border-emerald-500/30',
      icon: (
        <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
    },
    {
      label: 'P95 Latency',
      value: `${p95Latency} ms`,
      subtext: 'Tail latency SLA',
      color: 'from-purple-500/20 to-indigo-500/10 text-purple-400 border-purple-500/30',
      icon: (
        <svg className="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      ),
    },
  ];

  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-3.5 sm:gap-4">
      {cards.map((c, i) => (
        <div
          key={i}
          className={`relative overflow-hidden rounded-xl border bg-gradient-to-br ${c.color} bg-slate-900/60 p-4 transition-all duration-300 hover:scale-[1.01] hover:border-slate-700 backdrop-blur-sm`}
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-medium text-slate-400">{c.label}</span>
            <div className="p-1.5 rounded-lg bg-slate-800/80 border border-slate-700/50">
              {c.icon}
            </div>
          </div>
          <div className="mt-3">
            <div className="text-2xl font-bold tracking-tight text-white font-mono">
              {c.value}
            </div>
            <div className="text-[11px] text-slate-400 mt-1">{c.subtext}</div>
          </div>
        </div>
      ))}
    </div>
  );
};
