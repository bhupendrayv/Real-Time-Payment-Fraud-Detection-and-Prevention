import React, { useMemo, useState } from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { StatsResponse, TransactionItem } from '../types';

interface ScoreChartProps {
  transactions: TransactionItem[];
  stats: StatsResponse | null;
  threshold?: number;
}

export const ScoreChart: React.FC<ScoreChartProps> = ({
  transactions,
  stats,
  threshold = 0.835,
}) => {
  const [viewMode, setViewMode] = useState<'distribution' | 'timeline'>('distribution');

  // Compute 10 score buckets from recent transactions
  const distributionData = useMemo(() => {
    const buckets = [
      { range: '0.0-0.1', min: 0.0, max: 0.1, count: 0, flagged: 0 },
      { range: '0.1-0.2', min: 0.1, max: 0.2, count: 0, flagged: 0 },
      { range: '0.2-0.3', min: 0.2, max: 0.3, count: 0, flagged: 0 },
      { range: '0.3-0.4', min: 0.3, max: 0.4, count: 0, flagged: 0 },
      { range: '0.4-0.5', min: 0.4, max: 0.5, count: 0, flagged: 0 },
      { range: '0.5-0.6', min: 0.5, max: 0.6, count: 0, flagged: 0 },
      { range: '0.6-0.7', min: 0.6, max: 0.7, count: 0, flagged: 0 },
      { range: '0.7-0.8', min: 0.7, max: 0.8, count: 0, flagged: 0 },
      { range: '0.8-0.9', min: 0.8, max: 0.9, count: 0, flagged: 0 },
      { range: '0.9-1.0', min: 0.9, max: 1.01, count: 0, flagged: 0 },
    ];

    transactions.forEach((tx) => {
      const score = tx.fraud_score;
      const b = buckets.find((bucket) => score >= bucket.min && score < bucket.max);
      if (b) {
        b.count += 1;
        if (tx.is_fraud) {
          b.flagged += 1;
        }
      }
    });

    return buckets.map((b) => ({
      range: b.range,
      Legitimate: b.count - b.flagged,
      Flagged: b.flagged,
      total: b.count,
    }));
  }, [transactions]);

  // Use hourly counts from stats or fallback to recent transactions chronological view
  const timelineData: { time: string; Total: number; Flagged: number }[] = useMemo(() => {
    if (stats?.hourly_counts && stats.hourly_counts.length > 0) {
      return stats.hourly_counts.map((hc) => ({
        time: hc.hour.slice(11, 16) || hc.hour,
        Total: hc.total,
        Flagged: hc.flagged,
      }));
    }

    // Fallback: Group transactions in chronological order (up to 30 points)
    const reversed = [...transactions].reverse().slice(0, 30);
    return reversed.map((tx, idx) => ({
      time: `#${idx + 1}`,
      Total: 1,
      Flagged: tx.is_fraud ? 1 : 0,
    }));
  }, [stats, transactions]);

  const activeThreshold = transactions[0]?.threshold ?? threshold;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 backdrop-blur-sm">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-5">
        <div>
          <h2 className="text-base font-semibold text-white flex items-center gap-2">
            <span>Fraud Score Analytics</span>
            <span className="text-xs font-mono font-normal px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              Threshold: {activeThreshold.toFixed(3)}
            </span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time inference probability distribution &amp; decision boundaries
          </p>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center gap-1 bg-slate-800/80 p-1 rounded-lg border border-slate-700/60 self-start sm:self-auto text-xs">
          <button
            type="button"
            onClick={() => setViewMode('distribution')}
            className={`px-3 py-1 rounded-md font-medium transition-all ${
              viewMode === 'distribution'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Score Buckets
          </button>
          <button
            type="button"
            onClick={() => setViewMode('timeline')}
            className={`px-3 py-1 rounded-md font-medium transition-all ${
              viewMode === 'timeline'
                ? 'bg-indigo-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Volume &amp; Timeline
          </button>
        </div>
      </div>

      <div className="h-64 w-full">
        {viewMode === 'distribution' ? (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={distributionData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="range" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: '#f8fafc',
                }}
              />
              <Bar dataKey="Legitimate" fill="#3b82f6" radius={[4, 4, 0, 0]} stackId="a" />
              <Bar dataKey="Flagged" fill="#ef4444" radius={[4, 4, 0, 0]} stackId="a" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={timelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="colorFlagged" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 11 }} />
              <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '8px',
                  fontSize: '12px',
                  color: '#f8fafc',
                }}
              />
              <ReferenceLine
                y={activeThreshold}
                stroke="#f43f5e"
                strokeDasharray="4 4"
                label={{
                  value: `Threshold ${activeThreshold.toFixed(3)}`,
                  fill: '#f43f5e',
                  fontSize: 10,
                  position: 'insideTopRight',
                }}
              />
              <Area
                type="monotone"
                dataKey="Total"
                stroke="#6366f1"
                fillOpacity={1}
                fill="url(#colorTotal)"
              />
              <Area
                type="monotone"
                dataKey="Flagged"
                stroke="#ef4444"
                fillOpacity={1}
                fill="url(#colorFlagged)"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="flex items-center justify-center gap-6 mt-3 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-blue-500 inline-block" />
          <span>Legitimate (&lt; {activeThreshold.toFixed(3)})</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 rounded-sm bg-rose-500 inline-block" />
          <span>Flagged Fraud (&ge; {activeThreshold.toFixed(3)})</span>
        </div>
      </div>
    </div>
  );
};
