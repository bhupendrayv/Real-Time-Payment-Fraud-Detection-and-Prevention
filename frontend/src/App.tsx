import React, { useCallback, useEffect, useRef, useState } from 'react';
import type {
  DriftResponse,
  HealthResponse,
  StatsResponse,
  TransactionItem,
} from './types';
import { getDrift, getHealth, getStats, getTransactions } from './api/client';
import { Header } from './components/Header';
import { StatCards } from './components/StatCards';
import { DriftCard } from './components/DriftCard';
import { ScoreChart } from './components/ScoreChart';
import { TransactionTable } from './components/TransactionTable';
import { TransactionModal } from './components/TransactionModal';
import { ManualTester } from './components/ManualTester';
import { ErrorBanner } from './components/ErrorBanner';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [transactions, setTransactions] = useState<TransactionItem[]>([]);
  const [drift, setDrift] = useState<DriftResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [driftLoading, setDriftLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [driftError, setDriftError] = useState<string | null>(null);

  // UI state
  const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
  const [flaggedOnly, setFlaggedOnly] = useState<boolean>(false);
  const [limit, setLimit] = useState<number>(50);
  const [selectedTx, setSelectedTx] = useState<TransactionItem | null>(null);
  const [isTesterOpen, setIsTesterOpen] = useState<boolean>(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const isMountedRef = useRef(true);

  const fetchData = useCallback(
    async (isBackground = false) => {
      if (!isBackground) setLoading(true);
      try {
        const [healthRes, statsRes, txsRes] = await Promise.all([
          getHealth(),
          getStats(),
          getTransactions(limit, flaggedOnly),
        ]);

        if (isMountedRef.current) {
          setHealth(healthRes);
          setStats(statsRes);
          setTransactions(txsRes);
          setError(null);
          setLastUpdated(new Date());
        }
      } catch (err: unknown) {
        if (isMountedRef.current) {
          setError(
            err instanceof Error
              ? err.message
              : 'Failed to connect to FraudShield API'
          );
          setHealth(null);
        }
      } finally {
        if (isMountedRef.current && !isBackground) {
          setLoading(false);
        }
      }
    },
    [limit, flaggedOnly]
  );

  const fetchDrift = useCallback(async (isBackground = false) => {
    if (!isBackground) setDriftLoading(true);
    try {
      const res = await getDrift();
      if (isMountedRef.current) {
        setDrift(res);
        setDriftError(null);
      }
    } catch (err: unknown) {
      if (isMountedRef.current) {
        setDriftError(err instanceof Error ? err.message : 'Drift fetch failed');
      }
    } finally {
      if (isMountedRef.current && !isBackground) {
        setDriftLoading(false);
      }
    }
  }, []);

  // Initial load
  useEffect(() => {
    isMountedRef.current = true;
    fetchData();
    fetchDrift();
    return () => {
      isMountedRef.current = false;
    };
  }, [fetchData, fetchDrift]);

  // Polling loop for transactions & stats (every 2s when active)
  useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      fetchData(true);
    }, 2000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchData]);

  // Polling loop for drift (every 10s when active)
  useEffect(() => {
    if (!autoRefresh) return;
    const driftInterval = setInterval(() => {
      fetchDrift(true);
    }, 10000);
    return () => clearInterval(driftInterval);
  }, [autoRefresh, fetchDrift]);

  const handleToggleAutoRefresh = () => {
    setAutoRefresh((prev) => !prev);
  };

  const handleToggleFlaggedOnly = (val: boolean) => {
    setFlaggedOnly(val);
  };

  const handleChangeLimit = (newLimit: number) => {
    setLimit(newLimit);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Top Navbar */}
      <Header
        connected={health !== null && health.status === 'ok'}
        modelVersion={health?.model_version ?? null}
        autoRefresh={autoRefresh}
        onToggleAutoRefresh={handleToggleAutoRefresh}
        onOpenTester={() => setIsTesterOpen(true)}
        lastUpdated={lastUpdated}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
        {/* Error notice if API offline */}
        {error && (
          <ErrorBanner message={error} onRetry={() => { fetchData(false); fetchDrift(false); }} />
        )}

        {/* 1. Stat Cards */}
        <section aria-label="Key Performance Metrics">
          <StatCards stats={stats} loading={loading} />
        </section>

        {/* 2. Data & Score Drift Monitor */}
        <section aria-label="Data and Prediction Drift">
          <DriftCard drift={drift} loading={driftLoading} error={driftError} />
        </section>

        {/* 3. Visual Analytics */}
        <section aria-label="Inference Score Distribution">
          <ScoreChart
            transactions={transactions}
            stats={stats}
            threshold={transactions[0]?.threshold ?? 0.835}
          />
        </section>

        {/* 4. Transaction Stream Table */}
        <section aria-label="Audit Transaction Log">
          <TransactionTable
            transactions={transactions}
            loading={loading}
            flaggedOnly={flaggedOnly}
            onToggleFlaggedOnly={handleToggleFlaggedOnly}
            limit={limit}
            onChangeLimit={handleChangeLimit}
            onSelectTransaction={(tx) => setSelectedTx(tx)}
          />
        </section>
      </main>

      {/* Modals */}
      <TransactionModal
        transaction={selectedTx}
        onClose={() => setSelectedTx(null)}
      />

      <ManualTester
        isOpen={isTesterOpen}
        onClose={() => setIsTesterOpen(false)}
        onPredictionSuccess={() => { fetchData(true); fetchDrift(true); }}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-4 text-center text-xs text-slate-500">
        <p>
          FraudShield Enterprise AI Inference Platform &bull; FastAPI Backend &bull; XGBoost &amp; SHAP Explainer
        </p>
      </footer>
    </div>
  );
};

export default App;
