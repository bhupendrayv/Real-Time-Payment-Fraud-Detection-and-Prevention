import React from 'react';

interface HeaderProps {
  connected: boolean;
  modelVersion: string | null;
  autoRefresh: boolean;
  onToggleAutoRefresh: () => void;
  onOpenTester: () => void;
  lastUpdated: Date | null;
}

export const Header: React.FC<HeaderProps> = ({
  connected,
  modelVersion,
  autoRefresh,
  onToggleAutoRefresh,
  onOpenTester,
  lastUpdated,
}) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30 transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 p-0.5 shadow-lg shadow-indigo-500/20">
            <div className="h-full w-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <svg
                className="w-5 h-5 text-indigo-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                />
              </svg>
            </div>
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-xl font-bold bg-gradient-to-r from-white via-slate-100 to-slate-400 bg-clip-text text-transparent">
                FraudShield
              </h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-medium">
                Real-Time Inference
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Payment Fraud Detection &amp; Explainability System
            </p>
          </div>
        </div>

        {/* Right side controls: Health, model, refresh, tester */}
        <div className="flex flex-wrap items-center gap-3">
          {/* Health indicator */}
          <div
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-colors ${
              connected
                ? 'bg-emerald-950/40 border-emerald-800/60 text-emerald-300'
                : 'bg-rose-950/40 border-rose-800/60 text-rose-300'
            }`}
            title={connected ? 'Connected to FastAPI' : 'API unreachable'}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                connected
                  ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.8)] animate-pulse'
                  : 'bg-rose-500'
              }`}
            />
            <span>{connected ? 'API Online' : 'API Offline'}</span>
          </div>

          {/* Model version */}
          {modelVersion && (
            <div className="px-3 py-1.5 rounded-lg border border-slate-800 bg-slate-800/50 text-xs font-mono text-slate-300">
              <span className="text-slate-500 mr-1.5">Model:</span>
              <span className="font-semibold text-indigo-300">{modelVersion}</span>
            </div>
          )}

          {/* Auto-refresh toggle */}
          <button
            type="button"
            onClick={onToggleAutoRefresh}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
              autoRefresh
                ? 'bg-indigo-600/20 border-indigo-500/50 text-indigo-300 hover:bg-indigo-600/30'
                : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:bg-slate-800'
            }`}
            title={autoRefresh ? 'Auto-refresh every 2s is ON' : 'Auto-refresh is paused'}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                autoRefresh ? 'bg-indigo-400 animate-pulse' : 'bg-slate-500'
              }`}
            />
            <span>{autoRefresh ? 'Live (2s)' : 'Paused'}</span>
          </button>

          {/* Manual test action */}
          <button
            type="button"
            onClick={onOpenTester}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-md shadow-indigo-600/25 transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <svg
              className="w-3.5 h-3.5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
            <span>Test Transaction</span>
          </button>

          {lastUpdated && (
            <span className="text-[11px] text-slate-500 font-mono hidden xl:inline">
              Updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>
    </header>
  );
};
