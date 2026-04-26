import React, { useState } from 'react';
import { upgradeImage, fetchCtrPrediction } from '../services/api';

const COUNTRIES = ['US', 'ES', 'UK', 'DE', 'FR', 'IT', 'BR', 'MX', 'JP', 'CA'];
const OS_OPTS   = ['iOS', 'Android'];

const SEGMENT_COLORS = {
  'US-iOS':     '#6366f1', 'US-Android': '#818cf8',
  'ES-iOS':     '#f59e0b', 'ES-Android': '#fbbf24',
  'UK-iOS':     '#10b981', 'UK-Android': '#34d399',
  'DE-iOS':     '#ef4444', 'DE-Android': '#f87171',
  'FR-iOS':     '#8b5cf6', 'FR-Android': '#a78bfa',
};
const fallbackColors = ['#64748b','#94a3b8','#475569','#cbd5e1'];

function getColor(country, os, idx = 0) {
  return SEGMENT_COLORS[`${country}-${os}`] || fallbackColors[idx % fallbackColors.length];
}

// Pure SVG line chart
function LineChart({ series }) {
  if (!series || series.length === 0) return null;
  const W = 560, H = 200;
  const PAD = { top: 12, right: 12, bottom: 28, left: 44 };
  const cW = W - PAD.left - PAD.right;
  const cH = H - PAD.top  - PAD.bottom;

  const allVals = series.flatMap(s => s.values);
  const maxY = Math.max(...allVals, 0.001) * 1.15;
  const seqLen = series[0]?.values.length || 30;
  const xS = i => (i / (seqLen - 1)) * cW;
  const yS = v => cH - (v / maxY) * cH;

  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => t * maxY);
  const xLabels = [1, 7, 14, 21, 30].filter(d => d <= seqLen);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={0} y1={yS(t)} x2={cW} y2={yS(t)} stroke="#e2e8f0" strokeWidth={1} />
            <text x={-5} y={yS(t)+4} textAnchor="end" fontSize={9} fill="#94a3b8">
              {(t*100).toFixed(1)}%
            </text>
          </g>
        ))}
        {xLabels.map(d => (
          <text key={d} x={xS(d-1)} y={cH+16} textAnchor="middle" fontSize={9} fill="#94a3b8">D{d}</text>
        ))}
        {series.map(({ values, color, dashed }, si) => {
          const pts = values.map((v, i) => `${xS(i)},${yS(v)}`).join(' ');
          const peak = Math.max(...values);
          const pi   = values.indexOf(peak);
          return (
            <g key={si}>
              <polyline points={pts} fill="none" stroke={color}
                strokeWidth={dashed ? 1.5 : 2.5}
                strokeDasharray={dashed ? '5,4' : undefined}
                strokeLinecap="round" strokeLinejoin="round" />
              <circle cx={xS(pi)} cy={yS(peak)} r={3.5}
                fill={color} stroke="white" strokeWidth={1.5} />
            </g>
          );
        })}
      </g>
    </svg>
  );
}

const UPGRADE_STEPS = [
  'Retrieving top-performing creatives...',
  'Running GPT-4o Visual Enrichment...',
  'Identifying missing features...',
  'Running SAM mask generation...',
  'Stable Diffusion inpainting...',
  'Evaluating with LightGBM...',
];

export default function UpgradeModal({ creative, isOpen, onClose, onApply }) {
  const [isUpgrading, setIsUpgrading]   = useState(false);
  const [upgradedData, setUpgradedData] = useState(null);
  const [currentStep, setCurrentStep]   = useState(0);

  // Forecast state
  const [showForecast, setShowForecast]           = useState(false);
  const [selectedCountries, setSelectedCountries] = useState(['US', 'ES']);
  const [selectedOS, setSelectedOS]               = useState(['iOS', 'Android']);
  const [prediction, setPrediction]               = useState(null);
  const [predLoading, setPredLoading]             = useState(false);

  React.useEffect(() => {
    if (isOpen) {
      setUpgradedData(null);
      setIsUpgrading(false);
      setCurrentStep(0);
      setShowForecast(false);
      setPrediction(null);
    }
  }, [isOpen, creative?.id]);

  if (!isOpen || !creative) return null;

  const handleUpgrade = async () => {
    setIsUpgrading(true);
    setCurrentStep(0);
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => prev < UPGRADE_STEPS.length - 1 ? prev + 1 : prev);
    }, 2800);
    const result = await upgradeImage(creative.id);
    clearInterval(stepInterval);
    setUpgradedData(result);
    setIsUpgrading(false);
  };

  const handleApply = async () => {
    if (!upgradedData?.newImageUrl || !upgradedData?.creativeId) return;
    // Call /enrich in background — SAM+semantic, doesn't block UI
    try {
      fetch(`http://localhost:8000/api/creatives/${creative.id}/enrich`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          new_id: upgradedData.creativeId,
          image_url: upgradedData.newImageUrl,
        }),
      });
    } catch (_) { /* non-fatal */ }
    onApply(upgradedData.creativeId, upgradedData.newImageUrl);
  };

  const runForecast = async () => {
    setPredLoading(true);
    setPrediction(null);
    const result = await fetchCtrPrediction(creative.id, {
      countries: selectedCountries.join(','),
      os: selectedOS.join(','),
      compareImageUrl: upgradedData?.newImageUrl || null,
      seqLen: 30,
    });
    setPrediction(result);
    setPredLoading(false);
  };

  const toggleCountry = c => setSelectedCountries(prev =>
    prev.includes(c) ? (prev.length > 1 ? prev.filter(x => x !== c) : prev) : [...prev, c]
  );
  const toggleOS = o => setSelectedOS(prev =>
    prev.includes(o) ? (prev.length > 1 ? prev.filter(x => x !== o) : prev) : [...prev, o]
  );

  // Build series for chart (generated image only)
  const buildSeries = (data) => {
    if (!data?.predictions) return [];
    return data.predictions.map((p, i) => ({
      label: `${p.country} · ${p.os}`,
      values: p.ctr_timeseries,
      color: getColor(p.country, p.os, i),
      peak_ctr: p.peak_ctr,
      avg_ctr: p.avg_ctr,
      fatigue_day: p.fatigue_day,
      country: p.country,
      os: p.os,
    }));
  };

  const genSeries = buildSeries(prediction?.generated);

  return (
    <div className="fixed inset-0 bg-slate-900/90 backdrop-blur-xl z-[100] flex items-center justify-center p-0 md:p-6 animate-in fade-in duration-300">
      <div className="bg-white md:rounded-[3rem] shadow-2xl w-full h-full md:h-auto md:max-w-6xl md:max-h-[90vh] overflow-y-auto flex flex-col md:flex-row relative border border-white/20">

        {/* Left: Original */}
        <div className="w-full md:flex-1 bg-slate-50 p-6 md:p-10 flex flex-col justify-center border-b md:border-b-0 md:border-r border-slate-100">
          <div className="mb-6">
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-indigo-500 mb-2 block">Source Asset</span>
            <h3 className="text-2xl md:text-3xl font-black tracking-tighter uppercase italic">{creative.advertiser}</h3>
          </div>
          <div className="relative group rounded-[1.5rem] md:rounded-[2rem] overflow-hidden shadow-2xl shadow-slate-300 bg-slate-900 flex items-center justify-center">
            <img src={creative.image_url} alt="Original"
              className="w-full h-auto object-contain aspect-[9/16] max-h-[40vh] md:max-h-[55vh]" />
            <div className="absolute inset-0 bg-red-500/10 mix-blend-overlay pointer-events-none"></div>
          </div>
          <div className="mt-8 space-y-4">
            <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-slate-400 border-b border-slate-200 pb-2">
              <span>Performance</span>
              <span className={creative.fatigued ? 'text-red-500' : 'text-emerald-500'}>
                {creative.fatigued ? 'Fatigued' : `Score ${creative.performance_score}`}
              </span>
            </div>
            {creative.insights && (
              <p className="text-xs md:text-sm text-slate-500 leading-relaxed font-medium">"{ creative.insights}"</p>
            )}
          </div>
        </div>

        {/* Right: AI Upgrade / Forecast */}
        <div className="w-full md:flex-1 p-6 md:p-10 flex flex-col bg-white">
          <div className="mb-6 md:mb-8 flex items-center justify-between">
            <div>
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-emerald-500 mb-2 block">PixelForge Diffusion V4</span>
              <h3 className="text-2xl md:text-3xl font-black tracking-tighter uppercase italic">
                {showForecast ? 'CTR Forecast' : 'AI Reconstruction'}
              </h3>
            </div>
            {/* Toggle button — only visible after upgrade success */}
            {upgradedData?.success && (
              <button
                onClick={() => { setShowForecast(p => !p); if (!showForecast && !prediction) runForecast(); }}
                className={`px-4 py-2 rounded-2xl font-black text-[10px] uppercase tracking-widest transition-all ${
                  showForecast
                    ? 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    : 'bg-indigo-600 text-white shadow-lg shadow-indigo-200 hover:bg-indigo-700'
                }`}
              >
                {showForecast ? '← Back' : '📈 Forecast'}
              </button>
            )}
            {/* Close button — always rightmost */}
            <button
              onClick={onClose}
              className="w-9 h-9 bg-slate-100 hover:bg-slate-200 rounded-xl flex items-center justify-center text-slate-500 transition-all flex-shrink-0"
              aria-label="Close"
            >
              ✕
            </button>
          </div>

          {/* ── UPGRADE SLOT ── */}
          {!showForecast && (
            <div className="flex-1 flex flex-col items-center justify-center border-4 border-dashed border-slate-100 rounded-[1.5rem] md:rounded-[2rem] bg-slate-50/50 p-6 md:p-8 relative min-h-[300px]">
              {!isUpgrading && !upgradedData && (
                <div className="text-center group">
                  <div className="text-5xl md:text-7xl mb-6 group-hover:scale-110 transition-transform duration-500">✨</div>
                  <h4 className="text-base md:text-lg font-black text-slate-800 uppercase tracking-tighter mb-2">Ready for optimization</h4>
                  <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-8">Estimated compute: 3.2s</p>
                  <button onClick={handleUpgrade}
                    className="bg-indigo-600 hover:bg-emerald-500 text-white font-black px-8 md:px-12 py-4 md:py-5 rounded-xl md:rounded-[1.5rem] shadow-xl shadow-indigo-200 hover:shadow-emerald-200 transition-all active:scale-95 uppercase tracking-widest text-[10px] flex items-center gap-3 mx-auto">
                    Start AI Generation
                  </button>
                </div>
              )}

              {isUpgrading && (
                <div className="text-center w-full">
                  <div className="w-12 h-12 md:w-16 md:h-16 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin mx-auto mb-6"></div>
                  <h4 className="text-base md:text-lg font-black text-slate-800 italic animate-pulse tracking-tighter mb-6">RUNNING AI PIPELINE...</h4>
                  <div className="space-y-2 text-left max-w-xs mx-auto">
                    {UPGRADE_STEPS.map((step, i) => (
                      <div key={i} className={`flex items-center gap-3 text-[11px] font-bold transition-all ${
                        i < currentStep ? 'text-emerald-600' : i === currentStep ? 'text-indigo-600 animate-pulse' : 'text-slate-300'
                      }`}>
                        <span>{i < currentStep ? '✓' : i === currentStep ? '⟳' : '○'}</span>
                        {step}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {upgradedData && !upgradedData.success && (
                <div className="text-center">
                  <div className="text-5xl mb-4">⚠️</div>
                  <h4 className="text-base font-black text-red-600 mb-2">Upgrade Failed</h4>
                  <p className="text-xs text-slate-500 font-mono bg-slate-100 p-3 rounded-xl">{upgradedData.error}</p>
                  <button onClick={() => setUpgradedData(null)} className="mt-4 text-[10px] font-black text-indigo-500 uppercase tracking-widest hover:underline">Try Again</button>
                </div>
              )}

              {upgradedData?.success && (
                <div className="animate-in zoom-in duration-700 flex flex-col">
                  <div className="relative rounded-[1.5rem] md:rounded-[2rem] overflow-hidden shadow-2xl shadow-emerald-200 mb-6 md:mb-8 border-4 border-emerald-500 bg-slate-900">
                    <img
                      src={upgradedData.newImageUrl}
                      alt="Upgraded"
                      className="w-full object-contain" style={{ maxHeight: '55vh' }}
                    />
                    <div className="absolute top-4 left-4 bg-emerald-500 text-white text-[10px] px-3 py-1 rounded-full font-black uppercase tracking-widest shadow-lg">Optimized</div>
                  </div>
                  <div className="bg-emerald-50/50 p-4 md:p-6 rounded-2xl md:rounded-3xl border border-emerald-100 mb-4">
                    <div className="flex justify-between items-center mb-4">
                      <h5 className="text-[10px] font-black text-emerald-600 uppercase tracking-widest flex items-center gap-2">
                        <span className="w-1.5 h-1.5 bg-emerald-500 rounded-full"></span>
                        AI Logic Engine
                      </h5>
                      <div className="flex gap-2">
                        <div className="bg-slate-800 text-white text-[10px] font-black px-3 py-1 rounded-lg shadow-lg">Score: {upgradedData.performanceScore}</div>
                        <div className="bg-emerald-600 text-white text-[10px] font-black px-3 py-1 rounded-lg shadow-lg animate-pulse">Uplift: {upgradedData.predictedUplift}</div>
                      </div>
                    </div>
                    <p className="text-[11px] md:text-sm text-slate-600 font-medium leading-relaxed italic">"{upgradedData.aiReasoning}"</p>
                  </div>
                  <div className="mt-4 flex flex-col md:flex-row gap-3">
                    <button onClick={handleApply}
                      className="flex-1 bg-emerald-600 hover:bg-indigo-600 text-white font-black py-4 rounded-xl md:rounded-2xl shadow-xl shadow-emerald-100 transition-all active:scale-95 uppercase tracking-widest text-[10px]">
                      Replace Image
                    </button>
                    <button onClick={onClose}
                      className="flex-1 md:flex-none px-6 py-4 border border-slate-200 text-slate-400 font-black hover:bg-slate-50 rounded-xl md:rounded-2xl transition-all uppercase tracking-widest text-[10px]">
                      Keep Original
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── FORECAST SLOT ── fills the whole right space ── */}
          {showForecast && (
            <div className="flex-1 flex flex-col min-h-0">
              {/* Demographic controls */}
              <div className="flex flex-wrap gap-4 items-end mb-4 pb-4 border-b border-slate-100">
                <div>
                  <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 mb-1.5">Countries</p>
                  <div className="flex flex-wrap gap-1.5">
                    {COUNTRIES.map(c => (
                      <button key={c} onClick={() => toggleCountry(c)}
                        className={`px-2 py-0.5 rounded-lg text-[10px] font-black uppercase tracking-wide transition-all ${
                          selectedCountries.includes(c)
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                        }`}>{c}</button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 mb-1.5">Platform</p>
                  <div className="flex gap-1.5">
                    {OS_OPTS.map(o => (
                      <button key={o} onClick={() => toggleOS(o)}
                        className={`px-3 py-0.5 rounded-lg text-[10px] font-black uppercase tracking-wide transition-all ${
                          selectedOS.includes(o)
                            ? 'bg-indigo-600 text-white'
                            : 'bg-slate-100 text-slate-500 hover:bg-slate-200'
                        }`}>{o}</button>
                    ))}
                  </div>
                </div>
                <button onClick={runForecast} disabled={predLoading}
                  className="ml-auto px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white font-black text-[10px] uppercase tracking-widest shadow-lg shadow-indigo-200 transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2">
                  {predLoading
                    ? <><span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Running...</>
                    : '⚡ Run'}
                </button>
              </div>

              {/* Chart */}
              {predLoading && (
                <div className="flex-1 flex flex-col items-center justify-center">
                  <div className="w-10 h-10 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mb-4"></div>
                  <p className="text-slate-400 font-bold text-xs uppercase tracking-widest">Running neural inference...</p>
                </div>
              )}

              {!predLoading && !prediction && (
                <div className="flex-1 flex flex-col items-center justify-center text-center">
                  <div className="text-5xl mb-4">🧠</div>
                  <p className="text-slate-400 font-bold text-sm uppercase tracking-widest">Loading forecast...</p>
                </div>
              )}

              {!predLoading && prediction && (
                <div className="flex-1 flex flex-col">
                  {/* Legend */}
                  <div className="flex flex-wrap gap-3 mb-3">
                    {genSeries.map(s => (
                      <div key={s.label} className="flex items-center gap-1.5">
                        <span className="w-3 h-0.5 rounded-full inline-block" style={{ background: s.color, height: 3 }} />
                        <span className="text-[10px] font-bold text-slate-600">{s.label}</span>
                      </div>
                    ))}
                  </div>

                  {/* SVG chart — grows to fill space */}
                  <div className="flex-1 min-h-[180px]">
                    <LineChart series={genSeries} />
                  </div>

                  {/* Metrics table */}
                  <div className="mt-3 rounded-2xl border border-slate-100 overflow-hidden">
                    <table className="w-full text-[11px]">
                      <thead>
                        <tr className="bg-slate-50 border-b border-slate-100">
                          <th className="px-3 py-2 text-left font-black text-slate-400 uppercase tracking-widest text-[9px]">Segment</th>
                          <th className="px-3 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Peak CTR</th>
                          <th className="px-3 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Avg CTR</th>
                          <th className="px-3 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Fatigue</th>
                        </tr>
                      </thead>
                      <tbody>
                        {genSeries.map(seg => (
                          <tr key={seg.label} className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors">
                            <td className="px-3 py-2 font-bold text-slate-700 flex items-center gap-2">
                              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: seg.color }} />
                              {seg.country} · {seg.os}
                            </td>
                            <td className="px-3 py-2 text-right font-mono font-bold text-slate-700">{(seg.peak_ctr * 100).toFixed(2)}%</td>
                            <td className="px-3 py-2 text-right font-mono text-slate-500">{(seg.avg_ctr * 100).toFixed(2)}%</td>
                            <td className="px-3 py-2 text-right text-slate-500">{seg.fatigue_day ? `Day ${seg.fatigue_day}` : '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <p className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mt-2 text-center">
                    Predicted 30-day CTR for AI-generated creative · {prediction.generated?.summary.best_segment} is top segment
                  </p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
