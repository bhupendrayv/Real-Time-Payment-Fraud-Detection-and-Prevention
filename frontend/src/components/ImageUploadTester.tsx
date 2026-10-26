import React, { useCallback, useRef, useState } from 'react';
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
import { predictFromImage, type ImagePredictResponse } from '../api/client';

interface ImageUploadTesterProps {
  isOpen: boolean;
  onClose: () => void;
}

type UploadState = 'idle' | 'dragging' | 'loading' | 'done' | 'error';

export const ImageUploadTester: React.FC<ImageUploadTesterProps> = ({ isOpen, onClose }) => {
  const [uploadState, setUploadState] = useState<UploadState>('idle');
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [result, setResult] = useState<ImagePredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setUploadState('idle');
    setPreviewUrl(null);
    setFileName(null);
    setResult(null);
    setError(null);
  };

  const processFile = async (file: File) => {
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setFileName(file.name);
    setUploadState('loading');
    setResult(null);
    setError(null);

    try {
      const res = await predictFromImage(file, true);
      setResult(res);
      setUploadState('done');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
      setUploadState('error');
    }
  };

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG, PNG, or WEBP).');
      setUploadState('error');
      return;
    }
    processFile(file);
  };

  // useCallback must be declared before any conditional return (Rules of Hooks)
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setUploadState('idle');
    handleFiles(e.dataTransfer.files);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setUploadState('dragging');
  };

  const handleDragLeave = () => {
    if (uploadState === 'dragging') setUploadState('idle');
  };

  // All hooks declared above — safe to early-return now
  if (!isOpen) return null;


  const explanationData = (result?.explanation || []).map((exp) => ({
    feature: exp.feature,
    contribution: Number(exp.contribution.toFixed(4)),
  }));

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-6 overflow-y-auto">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-slate-950/85 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-2xl bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl z-10 overflow-hidden my-8 max-h-[92vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-gradient-to-r from-violet-950/50 to-slate-900">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-violet-600/20 border border-violet-500/30 flex items-center justify-center">
              <svg className="w-5 h-5 text-violet-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-semibold text-white">Payment Image Scanner</h3>
              <p className="text-xs text-slate-400 mt-0.5">Upload a receipt or payment screenshot to detect fraud</p>
            </div>
          </div>
          <button type="button" onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="p-6 overflow-y-auto space-y-5 flex-1">

          {/* Info banner */}
          <div className="flex items-start gap-3 bg-violet-950/30 border border-violet-800/40 rounded-xl p-3.5">
            <svg className="w-4 h-4 text-violet-400 mt-0.5 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-xs text-violet-300 leading-relaxed">
              Upload a <strong>payment receipt</strong>, <strong>bank screenshot</strong>, or <strong>transaction slip</strong>.
              The AI will extract the amount and timestamp, then run the fraud model on it.
              <br /><span className="opacity-70 mt-1 block">Supported: JPEG, PNG, WEBP · Max 10 MB</span>
            </p>
          </div>

          {/* Drop zone */}
          {(uploadState === 'idle' || uploadState === 'dragging' || uploadState === 'error') && (
            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onClick={() => inputRef.current?.click()}
              className={`
                relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer
                transition-all duration-200 group
                ${uploadState === 'dragging'
                  ? 'border-violet-500 bg-violet-950/40 scale-[1.01]'
                  : 'border-slate-700 hover:border-violet-600/60 hover:bg-violet-950/10'
                }
              `}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp"
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />
              <div className="flex flex-col items-center gap-3">
                <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-colors
                  ${uploadState === 'dragging' ? 'bg-violet-600/30' : 'bg-slate-800 group-hover:bg-violet-900/30'}`}>
                  <svg className={`w-8 h-8 transition-colors ${uploadState === 'dragging' ? 'text-violet-400' : 'text-slate-500 group-hover:text-violet-400'}`}
                    fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5"
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-300 group-hover:text-white transition-colors">
                    {uploadState === 'dragging' ? 'Drop your image here' : 'Click to upload or drag & drop'}
                  </p>
                  <p className="text-xs text-slate-500 mt-1">Payment receipt, bank notification, or transaction screenshot</p>
                </div>
                <div className="flex items-center gap-2 text-xs text-slate-600">
                  <span className="px-2 py-0.5 bg-slate-800 rounded">JPEG</span>
                  <span className="px-2 py-0.5 bg-slate-800 rounded">PNG</span>
                  <span className="px-2 py-0.5 bg-slate-800 rounded">WEBP</span>
                </div>
              </div>

              {uploadState === 'error' && error && (
                <div className="absolute bottom-3 left-3 right-3 p-2 bg-rose-950/70 border border-rose-800/60 rounded-lg text-xs text-rose-300">
                  {error}
                </div>
              )}
            </div>
          )}

          {/* Loading state */}
          {uploadState === 'loading' && (
            <div className="space-y-4">
              {previewUrl && (
                <div className="relative rounded-xl overflow-hidden border border-slate-700 max-h-52">
                  <img src={previewUrl} alt="Uploaded receipt" className="w-full h-52 object-contain bg-slate-950" />
                  <div className="absolute inset-0 bg-slate-950/70 flex flex-col items-center justify-center gap-3">
                    <div className="w-12 h-12 rounded-full border-4 border-violet-600/30 border-t-violet-500 animate-spin" />
                    <div className="text-center">
                      <p className="text-sm font-semibold text-white">Analyzing Payment Image...</p>
                      <p className="text-xs text-slate-400 mt-1">Extracting fields · Running fraud model · Computing SHAP</p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Result state */}
          {uploadState === 'done' && result && (
            <div className="space-y-4">
              {/* Image preview + extracted fields side by side */}
              <div className="grid grid-cols-2 gap-4">
                {previewUrl && (
                  <div className="rounded-xl overflow-hidden border border-slate-700 bg-slate-950">
                    <div className="text-[10px] text-slate-500 px-2 py-1 border-b border-slate-800 font-mono flex items-center gap-1">
                      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                      {fileName}
                    </div>
                    <img src={previewUrl} alt="Receipt" className="w-full h-36 object-contain p-2" />
                  </div>
                )}
                {/* Extracted fields */}
                <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 space-y-2">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider font-semibold mb-2">
                    {result.ocr_available ? '🔍 OCR Extracted Fields' : '📋 Fields Used for Analysis'}
                  </div>
                  {!result.ocr_available && (
                    <p className="text-[10px] text-amber-400/80 leading-snug">
                      Install Tesseract OCR to enable automatic field extraction from images.
                    </p>
                  )}
                  <ExtractedField label="Amount" value={result.extracted_fields.amount !== null ? `$${result.extracted_fields.amount?.toFixed(2)}` : `$${result.fields_used.Amount.toFixed(2)} (default)`} hasValue={result.extracted_fields.amount !== null} />
                  <ExtractedField label="Time" value={result.extracted_fields.time_seconds !== null ? `${result.extracted_fields.time_seconds}s` : `${result.fields_used.Time}s (default)`} hasValue={result.extracted_fields.time_seconds !== null} />
                  <ExtractedField label="Card" value={result.extracted_fields.card_last4 ? `****${result.extracted_fields.card_last4}` : 'Not detected'} hasValue={!!result.extracted_fields.card_last4} />
                  <ExtractedField label="Merchant" value={result.extracted_fields.merchant || 'Not detected'} hasValue={!!result.extracted_fields.merchant} />
                </div>
              </div>

              {/* Fraud decision card */}
              <div className={`p-4 rounded-xl border flex items-center justify-between ${
                result.is_fraud
                  ? 'bg-rose-950/40 border-rose-800/70 text-rose-200'
                  : 'bg-emerald-950/40 border-emerald-800/70 text-emerald-200'
              }`}>
                <div>
                  <div className="flex items-center gap-2">
                    <span className={`w-3 h-3 rounded-full ${result.is_fraud ? 'bg-rose-500 animate-pulse' : 'bg-emerald-400'}`} />
                    <span className="text-sm font-bold">
                      {result.is_fraud ? '🚨 FRAUD DETECTED' : '✅ PAYMENT APPEARS LEGITIMATE'}
                    </span>
                  </div>
                  <span className="text-xs opacity-70 block mt-1">
                    Threshold: {(result.threshold * 100).toFixed(1)}% · Model: {result.model_version} · {result.latency_ms.toFixed(1)}ms
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold font-mono">{(result.fraud_score * 100).toFixed(1)}%</div>
                  <div className="text-[10px] uppercase tracking-wider opacity-70">Fraud Score</div>
                </div>
              </div>

              {/* Fraud score bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-slate-500">
                  <span>Low Risk</span>
                  <span>Threshold ({(result.threshold * 100).toFixed(0)}%)</span>
                  <span>High Risk</span>
                </div>
                <div className="relative h-3 bg-slate-800 rounded-full overflow-hidden">
                  <div className="absolute inset-0 bg-gradient-to-r from-emerald-600 via-amber-500 to-rose-600 opacity-30 rounded-full" />
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${result.is_fraud ? 'bg-rose-500' : 'bg-emerald-500'}`}
                    style={{ width: `${result.fraud_score * 100}%` }}
                  />
                  {/* Threshold marker */}
                  <div
                    className="absolute top-0 bottom-0 w-0.5 bg-white/50"
                    style={{ left: `${result.threshold * 100}%` }}
                  />
                </div>
              </div>

              {/* SHAP chart */}
              {explanationData.length > 0 && (
                <div className="bg-slate-950/70 p-4 rounded-xl border border-slate-800">
                  <div className="text-xs font-semibold text-slate-300 mb-2">
                    Top Feature Attributions (SHAP) — Why did the model decide this?
                  </div>
                  <div className="h-44 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart layout="vertical" data={explanationData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis type="number" stroke="#64748b" tick={{ fontSize: 10 }} />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          stroke="#64748b"
                          tick={{ fontSize: 11, fill: '#cbd5e1' }}
                          width={65}
                        />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px', color: '#f8fafc' }}
                          formatter={(val: unknown) => [
                            `${Number(val) > 0 ? '+' : ''}${String(val)} (${Number(val) > 0 ? 'Pushes toward Fraud' : 'Pushes toward Legitimate'})`,
                            'SHAP Value',
                          ]}
                        />
                        <ReferenceLine x={0} stroke="#475569" />
                        <Bar dataKey="contribution" radius={[0, 4, 4, 0]}>
                          {explanationData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.contribution > 0 ? '#ef4444' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="flex justify-center gap-6 mt-2 text-[10px] text-slate-400">
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-rose-500 inline-block" /> Increases fraud risk</span>
                    <span className="flex items-center gap-1"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-500 inline-block" /> Decreases fraud risk</span>
                  </div>
                </div>
              )}

              {/* OCR raw text (collapsed) */}
              {result.ocr_available && result.ocr_text && (
                <details className="bg-slate-950/60 border border-slate-800 rounded-xl">
                  <summary className="px-4 py-2.5 text-xs text-slate-400 cursor-pointer hover:text-slate-300 select-none">
                    🔍 View Raw OCR Text
                  </summary>
                  <pre className="px-4 pb-3 text-[10px] text-slate-400 font-mono whitespace-pre-wrap leading-relaxed max-h-32 overflow-y-auto">
                    {result.ocr_text}
                  </pre>
                </details>
              )}

              {/* Try another */}
              <button
                type="button"
                onClick={reset}
                className="w-full py-2 rounded-xl border border-slate-700 hover:bg-slate-800 text-xs text-slate-300 font-semibold transition-colors flex items-center justify-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                </svg>
                Upload Another Image
              </button>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-slate-800 bg-slate-900/90 flex items-center justify-between">
          <span className="text-[10px] text-slate-600">
            POST /predict/from-image · SHAP explanations enabled
          </span>
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

// Helper component
const ExtractedField: React.FC<{ label: string; value: string; hasValue: boolean }> = ({ label, value, hasValue }) => (
  <div className="flex items-center justify-between gap-2">
    <span className="text-[10px] text-slate-500 shrink-0">{label}</span>
    <span className={`text-[11px] font-mono truncate ${hasValue ? 'text-emerald-400' : 'text-slate-600'}`}>
      {value}
    </span>
  </div>
);
