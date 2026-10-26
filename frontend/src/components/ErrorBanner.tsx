import React from 'react';

interface ErrorBannerProps {
  message: string;
  onRetry: () => void;
}

export const ErrorBanner: React.FC<ErrorBannerProps> = ({ message, onRetry }) => {
  return (
    <div className="rounded-xl border border-rose-800/80 bg-rose-950/40 p-4 text-rose-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 shadow-lg shadow-rose-950/30">
      <div className="flex items-start sm:items-center gap-3">
        <div className="p-2 rounded-lg bg-rose-900/60 border border-rose-700/60 text-rose-300 mt-0.5 sm:mt-0">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
            />
          </svg>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-rose-100">
            FastAPI Connection Notice
          </h4>
          <p className="text-xs text-rose-300 mt-0.5">
            {message}
          </p>
          <span className="text-[11px] text-rose-400 font-mono block mt-1">
            Tip: Run <code className="bg-rose-950 px-1 py-0.5 rounded border border-rose-800">python -m uvicorn api.main:app --host 127.0.0.1 --port 8000</code>
          </span>
        </div>
      </div>

      <button
        type="button"
        onClick={onRetry}
        className="self-start sm:self-auto px-3.5 py-1.5 rounded-lg bg-rose-800 hover:bg-rose-700 text-white text-xs font-semibold transition-colors flex items-center gap-1.5"
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="2"
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
        <span>Retry Connection</span>
      </button>
    </div>
  );
};
