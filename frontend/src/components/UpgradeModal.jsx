import React, { useState, useEffect } from 'react';
import { upgradeImage, fetchCtrPrediction } from '../services/api';

// Same hook as CreativeCard — lazily fetches visual_semantic global description
function useCreativeDescription(creativeId) {
  const [description, setDescription] = useState(null);
  useEffect(() => {
    if (!creativeId) return;
    fetch(`/data/visual_semantic/creative_${creativeId}.json`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data?.global?.description) setDescription(data.global.description); })
      .catch(() => {});
  }, [creativeId]);
  return description;
}

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

// Pure SVG line chart — with fatigue markers
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
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full" preserveAspectRatio="none">
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
        {series.map(({ values, color, dashed, fatigue_index, fatigue_val }, si) => {
          const pts = values.map((v, i) => `${xS(i)},${yS(v)}`).join(' ');
          const peak = Math.max(...values);
          const pi   = values.indexOf(peak);
          return (
            <g key={si}>
              {fatigue_index !== null && fatigue_index !== undefined && (
                <g>
                  <line
                    x1={xS(fatigue_index)} y1={0}
                    x2={xS(fatigue_index)} y2={cH}
                    stroke="#ef4444" strokeWidth={1} strokeDasharray="4,4"
                  />
                  <circle cx={xS(fatigue_index)} cy={yS(fatigue_val)} r={4} fill="#ef4444" stroke="white" strokeWidth={1} />
                  <text x={xS(fatigue_index) - 6} y={cH - (dashed ? 30 : 10)} textAnchor="end" fontSize={9} fill="#ef4444" fontWeight="bold">Fatigue D{fatigue_index + 1}</text>
                </g>
              )}
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
  const [quality, setQuality]           = useState('medium'); // low | medium | high

  // Semantic description (same source as CreativeCard tooltip)
  const semanticDesc = useCreativeDescription(creative?.id);
  const displayDescription = semanticDesc || creative?.insights || null;

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
    const QUALITY_STEPS    = { low: 2, medium: 5, high: 15 };
    const QUALITY_STRENGTH = { low: 0.25, medium: 0.35, high: 0.50 };
    const numSteps = QUALITY_STEPS[quality] || 5;
    const strength = QUALITY_STRENGTH[quality] || 0.35;
    setIsUpgrading(true);
    setCurrentStep(0);
    const stepInterval = setInterval(() => {
      setCurrentStep(prev => prev < UPGRADE_STEPS.length - 1 ? prev + 1 : prev);
    }, 2800);
    const result = await upgradeImage(creative.id, { numSteps, strength });
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

  // Build series for chart — Original (dashed) vs AI Generated (solid)
  const buildComparisonSeries = (predObj) => {
    if (!predObj?.original?.predictions) return [];
    const all = [];

    // Map Original
    predObj.original.predictions.forEach((p, i) => {
      const fatigue_day = p.fatigue_day ? Math.floor(p.fatigue_day) : null;
      const fatigue_idx = fatigue_day ? fatigue_day - 1 : null;
      all.push({
        label: `${p.country} · ${p.os} (Orig)`,
        values: p.ctr_timeseries,
        color: getColor(p.country, p.os, i),
        dashed: true,
        peak_ctr: p.peak_ctr,
        avg_ctr: p.avg_ctr,
        fatigue_index: fatigue_idx,
        fatigue_val: fatigue_idx !== null && p.ctr_timeseries[fatigue_idx] !== undefined ? p.ctr_timeseries[fatigue_idx] : null,
        fatigue_day: fatigue_day,
        country: p.country,
        os: p.os,
        type: 'Original',
      });
    });

    // Map Upgraded / Generated
    if (predObj.generated?.predictions) {
      predObj.generated.predictions.forEach((p, i) => {
        const fatigue_day = p.fatigue_day ? Math.floor(p.fatigue_day) : null;
        const fatigue_idx = fatigue_day ? fatigue_day - 1 : null;
        all.push({
          label: `${p.country} · ${p.os} (AI)`,
          values: p.ctr_timeseries,
          color: getColor(p.country, p.os, i),
          dashed: false,
          peak_ctr: p.peak_ctr,
          avg_ctr: p.avg_ctr,
          fatigue_index: fatigue_idx,
          fatigue_val: fatigue_idx !== null && p.ctr_timeseries[fatigue_idx] !== undefined ? p.ctr_timeseries[fatigue_idx] : null,
          fatigue_day: fatigue_day,
          country: p.country,
          os: p.os,
          type: 'Upgraded',
        });
      });
    }

    return all;
  };

  const genSeries = buildComparisonSeries(prediction);

  const IMG_H = 'h-[42vh]';  // shared image slot height – both columns identical

  return (
    <div className="fixed inset-0 bg-slate-900/90 backdrop-blur-xl z-[100] flex items-center justify-center p-0 md:p-6 animate-in fade-in duration-300">
      <div className="bg-white md:rounded-[2.5rem] shadow-2xl w-full h-full md:h-[90vh] md:max-w-6xl overflow-hidden flex flex-col border border-white/20">

        {/* ── Header bar ── */}
        <div className="flex items-center justify-between px-6 md:px-10 py-4 border-b border-slate-100 flex-shrink-0">
          <div>
            <span className="text-[9px] font-black uppercase tracking-[0.25em] text-indigo-500">AD-EX · Creative Upgrade</span>
            <h2 className="text-lg font-black tracking-tighter uppercase italic leading-none">{creative.advertiser}</h2>
          </div>
          <div className="flex items-center gap-2">
            {upgradedData?.success && (
              <button
                onClick={() => { setShowForecast(p => !p); if (!showForecast && !prediction) runForecast(); }}
                className={`px-4 py-2 rounded-xl font-black text-[10px] uppercase tracking-widest transition-all ${
                  showForecast
                    ? 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                    : 'bg-indigo-600 text-white shadow-lg shadow-indigo-200 hover:bg-indigo-700'
                }`}
              >
                {showForecast ? '← Back' : '📈 Forecast'}
              </button>
            )}
            <button onClick={onClose}
              className="w-9 h-9 bg-slate-100 hover:bg-slate-200 rounded-xl flex items-center justify-center text-slate-500 transition-all"
              aria-label="Close">✕</button>
          </div>
        </div>

        {/* ── Body: two equal columns ── */}
        <div className="flex-1 flex flex-col md:flex-row overflow-hidden min-h-0">

          {/* Left: Original */}
          <div className="md:flex-1 bg-slate-50 flex flex-col border-b md:border-b-0 md:border-r border-slate-100 overflow-hidden">
            <div className="px-6 md:px-8 pt-5 pb-2 flex-shrink-0">
              <span className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 block mb-0.5">Original</span>
              <div className="flex justify-between items-center">
                <span className="text-xs font-black text-slate-700 uppercase tracking-tight">{creative.format}</span>
                {creative.fatigued ? (
                  <span className="text-[10px] font-black uppercase tracking-widest text-red-500">⚠ Fatigued • Needs Modify</span>
                ) : creative.performance_score < 0.4 ? (
                  <span className="text-[10px] font-black uppercase tracking-widest text-amber-500">⚠ Low Score • Needs Modify</span>
                ) : (
                  <span className="text-[10px] font-black uppercase tracking-widest text-emerald-500">✓ Healthy</span>
                )}
              </div>
            </div>
            {/* Image — fixed height, no overflow */}
            <div className={`mx-4 md:mx-6 rounded-2xl overflow-hidden bg-slate-900 relative flex-shrink-0 ${IMG_H}`}>
              <img src={creative.image_url} alt="Original"
                className="w-full h-full object-contain" />
              <div className="absolute inset-0 bg-red-500/8 mix-blend-overlay pointer-events-none" />
            </div>
            {/* Stats below image */}
            <div className="px-6 md:px-8 py-4 text-xs text-slate-500 leading-relaxed flex-1 overflow-y-auto">
              <div className="flex gap-2 flex-wrap mb-2">
                {[creative.theme, creative.hook_type].filter(Boolean).map(t => (
                  <span key={t} className="px-2 py-0.5 bg-slate-200 rounded-full text-[10px] font-bold text-slate-500 uppercase tracking-wider">{t}</span>
                ))}
              </div>
              {displayDescription && <p className="italic text-slate-400 line-clamp-3">"{displayDescription}"</p>}
            </div>
          </div>

          {/* Right: AI Upgrade / Forecast */}
          <div className="md:flex-1 flex flex-col bg-white overflow-hidden">
            <div className="px-6 md:px-8 pt-5 pb-2 flex-shrink-0">
              <span className="text-[9px] font-black uppercase tracking-[0.2em] text-emerald-500 block mb-0.5">
                {showForecast ? 'CTR Forecast' : 'AI Generated'}
              </span>
              <span className="text-xs font-black text-slate-700 uppercase tracking-tight">PixelForge Diffusion V4</span>
            </div>

            {/* ── UPGRADE SLOT – same fixed height as left image ── */}
            {!showForecast && (
              <>
                <div className={`mx-4 md:mx-6 rounded-2xl overflow-hidden bg-slate-100 border-2 border-dashed border-slate-200 flex-shrink-0 ${IMG_H} flex items-center justify-center relative`}>

                  {!isUpgrading && !upgradedData && (
                    <div className="text-center px-4">
                      <div className="text-5xl mb-4">✨</div>
                      <p className="text-[10px] text-slate-400 font-bold uppercase tracking-widest mb-4">Ready for optimization</p>
                      {/* Quality selector */}
                      <div className="flex justify-center gap-1.5 mb-5">
                        {[
                          { key: 'low',    label: 'Fast',   sub: '2 steps'  },
                          { key: 'medium', label: 'Balanced', sub: '5 steps' },
                          { key: 'high',   label: 'Detail', sub: '15 steps' },
                        ].map(({ key, label, sub }) => (
                          <button key={key} onClick={() => setQuality(key)}
                            className={`px-3 py-2 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all flex flex-col items-center ${
                              quality === key
                                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-200'
                                : 'bg-slate-100 text-slate-400 hover:bg-slate-200'
                            }`}>
                            <span>{label}</span>
                            <span className={`text-[8px] font-bold mt-0.5 ${quality === key ? 'text-indigo-200' : 'text-slate-300'}`}>{sub}</span>
                          </button>
                        ))}
                      </div>
                      <button onClick={handleUpgrade}
                        className="bg-indigo-600 hover:bg-emerald-500 text-white font-black px-8 py-3 rounded-xl shadow-xl shadow-indigo-200 transition-all active:scale-95 uppercase tracking-widest text-[10px] flex items-center gap-2 mx-auto">
                        Start AI Generation
                      </button>
                    </div>
                  )}

                  {isUpgrading && (
                    <div className="text-center w-full px-4">
                      <div className="w-12 h-12 border-4 border-indigo-100 border-t-indigo-600 rounded-full animate-spin mx-auto mb-4" />
                      <h4 className="text-sm font-black text-slate-800 italic animate-pulse tracking-tighter mb-4">RUNNING AI PIPELINE...</h4>
                      <div className="space-y-1.5 text-left max-w-[200px] mx-auto">
                        {UPGRADE_STEPS.map((step, i) => (
                          <div key={i} className={`flex items-center gap-2 text-[10px] font-bold transition-all ${
                            i < currentStep ? 'text-emerald-600' : i === currentStep ? 'text-indigo-600 animate-pulse' : 'text-slate-300'
                          }`}>
                            <span>{i < currentStep ? '✓' : i === currentStep ? '⟳' : '○'}</span>{step}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {upgradedData && !upgradedData.success && (
                    <div className="text-center px-4">
                      <div className="text-4xl mb-3">⚠️</div>
                      <p className="text-xs font-black text-red-600 mb-2">Upgrade Failed</p>
                      <p className="text-[10px] text-slate-500 font-mono bg-slate-100 p-2 rounded-lg">{upgradedData.error}</p>
                      <button onClick={() => setUpgradedData(null)} className="mt-3 text-[10px] font-black text-indigo-500 uppercase tracking-widest hover:underline">Try Again</button>
                    </div>
                  )}

                  {upgradedData?.success && (
                    <>
                      <img src={upgradedData.newImageUrl} alt="Upgraded"
                        className="w-full h-full object-contain animate-in zoom-in duration-700" />
                      <div className="absolute top-3 left-3 bg-emerald-500 text-white text-[10px] px-3 py-1 rounded-full font-black uppercase tracking-widest shadow-lg">Optimized</div>
                    </>
                  )}
                </div>

                {/* Below-image stats + actions */}
                <div className="px-6 md:px-8 py-4 flex-1 overflow-y-auto">
                  {upgradedData?.success ? (
                    <>
                      <div className="bg-emerald-50 border border-emerald-100 rounded-2xl p-4 mb-4">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-[10px] font-black text-emerald-600 uppercase tracking-widest">AI Reasoning</span>
                          <div className="flex items-center gap-2">
                            <span className="bg-slate-800 text-white text-[10px] font-black px-2 py-0.5 rounded-lg flex items-center gap-1">
                              <span className="text-slate-400 font-normal">Score:</span> {upgradedData.performanceScore}
                            </span>
                            <span className="bg-emerald-600 text-white text-[10px] font-black px-2 py-0.5 rounded-lg animate-pulse flex items-center gap-1">
                              <span className="text-emerald-200 font-normal">Uplift:</span> ↑{upgradedData.predictedUplift}
                            </span>
                          </div>
                        </div>
                        <p className="text-[11px] text-slate-600 leading-relaxed italic line-clamp-3">"{upgradedData.aiReasoning}"</p>
                      </div>
                      <div className="flex gap-2">
                        <button onClick={handleApply}
                          className="flex-1 bg-emerald-600 hover:bg-indigo-600 text-white font-black py-3 rounded-xl shadow-lg transition-all active:scale-95 uppercase tracking-widest text-[10px]">
                          Replace Image
                        </button>
                        <button onClick={onClose}
                          className="px-5 py-3 border border-slate-200 text-slate-400 font-black hover:bg-slate-50 rounded-xl transition-all uppercase tracking-widest text-[10px]">
                          Keep Original
                        </button>
                      </div>
                    </>
                  ) : (
                    <p className="text-[10px] text-slate-300 font-bold uppercase tracking-widest text-center mt-4">Awaiting generation...</p>
                  )}
                </div>
              </>
            )}

          {/* ── FORECAST SLOT ── fills the whole right space ── */}
          {showForecast && (
            <div className="flex-1 flex flex-col min-h-0 px-6 md:px-8 py-4 overflow-y-auto">
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
                        <span className="w-4 rounded-full inline-block" style={{
                          height: s.dashed ? 1 : 3,
                          borderTop: s.dashed ? `1px dashed ${s.color}` : 'none',
                          background: s.dashed ? 'transparent' : s.color,
                        }} />
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
                          <th className="px-3 py-2 text-center font-black text-slate-400 uppercase tracking-widest text-[9px]">Version</th>
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
                            <td className="px-3 py-2 text-center text-[9px] font-bold uppercase tracking-wide">
                              <span className={`px-2 py-0.5 rounded-md ${seg.type === 'Upgraded' ? 'bg-indigo-100 text-indigo-700' : 'bg-slate-100 text-slate-500'}`}>
                                {seg.type}
                              </span>
                            </td>
                            <td className="px-3 py-2 text-right font-mono font-bold text-slate-700">{(seg.peak_ctr * 100).toFixed(2)}%</td>
                            <td className="px-3 py-2 text-right font-mono text-slate-500">{(seg.avg_ctr * 100).toFixed(2)}%</td>
                            <td className="px-3 py-2 text-right">
                              {seg.fatigue_day ? (
                                <span className="text-red-500 font-bold bg-red-50 px-2 py-0.5 rounded">Day {seg.fatigue_day}</span>
                              ) : (
                                <span className="text-slate-400">—</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <p className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mt-2 text-center">
                    Predicted 30-day CTR comparison · {prediction.generated?.summary.best_segment || prediction.original?.summary.best_segment} is top{prediction.generated ? ' AI' : ''} segment
                  </p>
                </div>
              )}
            </div>
          )}
        </div> {/* end right column */}
        </div> {/* end body row */}
      </div> {/* end modal card */}
    </div>
  );
}
