import React, { useState, useEffect, useRef } from 'react';
import { fetchCtrPrediction } from '../services/api';

const COUNTRIES = ['US', 'ES', 'UK', 'DE', 'FR', 'IT', 'BR', 'MX', 'JP', 'CA'];
const OS_OPTS   = ['iOS', 'Android'];

// Color palette per segment
const SEGMENT_COLORS = {
  'US-iOS':     { stroke: '#6366f1', fill: '#6366f120' },
  'US-Android': { stroke: '#818cf8', fill: '#818cf820' },
  'ES-iOS':     { stroke: '#f59e0b', fill: '#f59e0b20' },
  'ES-Android': { stroke: '#fbbf24', fill: '#fbbf2420' },
  'UK-iOS':     { stroke: '#10b981', fill: '#10b98120' },
  'UK-Android': { stroke: '#34d399', fill: '#34d39920' },
  'DE-iOS':     { stroke: '#ef4444', fill: '#ef444420' },
  'DE-Android': { stroke: '#f87171', fill: '#f8717120' },
  'FR-iOS':     { stroke: '#8b5cf6', fill: '#8b5cf620' },
  'FR-Android': { stroke: '#a78bfa', fill: '#a78bfa20' },
};
const FALLBACK_COLOR = { stroke: '#94a3b8', fill: '#94a3b820' };

function getColor(country, os) {
  return SEGMENT_COLORS[`${country}-${os}`] || FALLBACK_COLOR;
}

// Pure SVG line chart — no external deps
function LineChart({ series, width = 600, height = 220 }) {
  if (!series || series.length === 0) return null;

  const PAD = { top: 16, right: 16, bottom: 32, left: 48 };
  const W = width - PAD.left - PAD.right;
  const H = height - PAD.top - PAD.bottom;

  const allValues = series.flatMap(s => s.values);
  const maxY = Math.max(...allValues, 0.001) * 1.15;
  const minY = 0;
  const seqLen = series[0]?.values.length || 30;

  const xScale = i => (i / (seqLen - 1)) * W;
  const yScale = v => H - ((v - minY) / (maxY - minY)) * H;

  // Y axis ticks
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map(t => minY + t * (maxY - minY));

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
      <g transform={`translate(${PAD.left},${PAD.top})`}>
        {/* Grid lines */}
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={0} y1={yScale(t)} x2={W} y2={yScale(t)} stroke="#e2e8f0" strokeWidth={1} />
            <text x={-6} y={yScale(t) + 4} textAnchor="end" fontSize={10} fill="#94a3b8">
              {(t * 100).toFixed(1)}%
            </text>
          </g>
        ))}

        {/* X axis ticks */}
        {[1, 7, 14, 21, 30].filter(d => d <= seqLen).map(d => (
          <text key={d} x={xScale(d - 1)} y={H + 18} textAnchor="middle" fontSize={10} fill="#94a3b8">
            D{d}
          </text>
        ))}

        {/* Area + Lines */}
        {series.map(({ label, values, color, dashed }) => {
          const pts = values.map((v, i) => `${xScale(i)},${yScale(v)}`).join(' ');
          const areaPath = [
            `M ${xScale(0)},${yScale(values[0])}`,
            ...values.map((v, i) => `L ${xScale(i)},${yScale(v)}`),
            `L ${xScale(values.length - 1)},${H}`,
            `L 0,${H} Z`,
          ].join(' ');

          return (
            <g key={label}>
              <path d={areaPath} fill={color.fill} />
              <polyline
                points={pts}
                fill="none"
                stroke={color.stroke}
                strokeWidth={dashed ? 1.5 : 2.5}
                strokeDasharray={dashed ? '5,4' : undefined}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              {/* Peak marker */}
              {(() => {
                const peak = Math.max(...values);
                const pi = values.indexOf(peak);
                return (
                  <circle cx={xScale(pi)} cy={yScale(peak)} r={4}
                    fill={color.stroke} stroke="white" strokeWidth={1.5} />
                );
              })()}
            </g>
          );
        })}
      </g>
    </svg>
  );
}

function MetricBadge({ label, value, color = 'indigo', sub }) {
  const colors = {
    indigo: 'bg-indigo-50 text-indigo-700 border-indigo-100',
    amber:  'bg-amber-50  text-amber-700  border-amber-100',
    green:  'bg-emerald-50 text-emerald-700 border-emerald-100',
    red:    'bg-red-50    text-red-700    border-red-100',
    slate:  'bg-slate-50  text-slate-600  border-slate-200',
  };
  return (
    <div className={`rounded-2xl border px-4 py-3 ${colors[color] || colors.slate}`}>
      <p className="text-[9px] font-black uppercase tracking-widest opacity-60 mb-0.5">{label}</p>
      <p className="text-lg font-black leading-none">{value}</p>
      {sub && <p className="text-[10px] opacity-60 mt-0.5">{sub}</p>}
    </div>
  );
}

export default function CtrPredictionPanel({ creative, upgradeImageUrl = null }) {
  const [selectedCountries, setSelectedCountries] = useState(['US', 'ES']);
  const [selectedOS, setSelectedOS] = useState(['iOS', 'Android']);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const abortRef = useRef(null);

  const runPrediction = async () => {
    if (abortRef.current) abortRef.current = false;
    setLoading(true);
    setError(null);
    setPrediction(null);

    const result = await fetchCtrPrediction(creative.id, {
      countries: selectedCountries.join(','),
      os: selectedOS.join(','),
      compareImageUrl: upgradeImageUrl || null,
      seqLen: 30,
    });

    if (!result) {
      setError('Prediction failed — check backend logs.');
    } else {
      setPrediction(result);
    }
    setLoading(false);
  };

  const toggleCountry = (c) => setSelectedCountries(prev =>
    prev.includes(c) ? (prev.length > 1 ? prev.filter(x => x !== c) : prev) : [...prev, c]
  );
  const toggleOS = (o) => setSelectedOS(prev =>
    prev.includes(o) ? (prev.length > 1 ? prev.filter(x => x !== o) : prev) : [...prev, o]
  );

  // Build chart series from API response
  const buildSeries = (data, dashed = false) => {
    if (!data?.predictions) return [];
    return data.predictions.map(p => ({
      label: `${data.label ?? ''} ${p.country} ${p.os}`.trim(),
      values: p.ctr_timeseries,
      color: getColor(p.country, p.os),
      dashed,
      country: p.country,
      os: p.os,
      peak_ctr: p.peak_ctr,
      avg_ctr: p.avg_ctr,
      fatigue_day: p.fatigue_day,
    }));
  };

  const origSeries = buildSeries(prediction?.original, false);
  const genSeries  = buildSeries(prediction?.generated, true);
  const allSeries  = [...origSeries, ...genSeries];

  const hasGenerated = prediction?.generated != null;

  return (
    <div className="mt-6 bg-white rounded-3xl border border-slate-100 shadow-xl shadow-slate-100 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-900 to-indigo-950 px-6 py-4 flex items-center justify-between">
        <div>
          <p className="text-[9px] font-black uppercase tracking-[0.3em] text-indigo-300">RNN Performance Forecast</p>
          <h3 className="text-base font-black text-white tracking-tight mt-0.5">30-Day CTR Prediction</h3>
        </div>
        <span className="text-xl">📈</span>
      </div>

      {/* Controls */}
      <div className="px-6 py-4 border-b border-slate-100 flex flex-wrap gap-4 items-end bg-slate-50/50">
        {/* Country selector */}
        <div>
          <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 mb-2">Countries</p>
          <div className="flex flex-wrap gap-1.5">
            {COUNTRIES.map(c => (
              <button key={c} onClick={() => toggleCountry(c)}
                className={`px-2.5 py-1 rounded-lg text-[10px] font-black uppercase tracking-wide transition-all ${
                  selectedCountries.includes(c)
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'bg-white border border-slate-200 text-slate-500 hover:border-indigo-300'
                }`}>
                {c}
              </button>
            ))}
          </div>
        </div>

        {/* OS selector */}
        <div>
          <p className="text-[9px] font-black uppercase tracking-widest text-slate-400 mb-2">Platform</p>
          <div className="flex gap-1.5">
            {OS_OPTS.map(o => (
              <button key={o} onClick={() => toggleOS(o)}
                className={`px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-wide transition-all ${
                  selectedOS.includes(o)
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'bg-white border border-slate-200 text-slate-500 hover:border-indigo-300'
                }`}>
                {o}
              </button>
            ))}
          </div>
        </div>

        <button onClick={runPrediction} disabled={loading}
          className="ml-auto px-5 py-2 rounded-2xl bg-indigo-600 hover:bg-indigo-700 text-white font-black text-[10px] uppercase tracking-widest shadow-lg shadow-indigo-200 transition-all active:scale-95 disabled:opacity-50 flex items-center gap-2">
          {loading ? (
            <>
              <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Running RNN...
            </>
          ) : '⚡ Predict CTR'}
        </button>
      </div>

      {/* Chart area */}
      {error && (
        <div className="px-6 py-8 text-center text-red-500 font-bold text-sm">{error}</div>
      )}

      {!prediction && !loading && !error && (
        <div className="px-6 py-12 text-center">
          <div className="text-5xl mb-4">🧠</div>
          <p className="text-slate-400 font-bold text-sm uppercase tracking-widest">
            Select demographics and click Predict CTR
          </p>
          <p className="text-slate-300 text-xs mt-2">
            The RNN will forecast 30-day CTR per country × platform segment
          </p>
        </div>
      )}

      {loading && (
        <div className="px-6 py-12 text-center">
          <div className="w-10 h-10 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto mb-4" />
          <p className="text-slate-500 font-bold text-sm uppercase tracking-widest">Running neural inference...</p>
        </div>
      )}

      {prediction && !loading && (
        <div className="px-6 py-5">
          {/* Legend */}
          <div className="flex flex-wrap gap-3 mb-4">
            {allSeries.map(s => (
              <div key={s.label} className="flex items-center gap-1.5">
                <svg width={28} height={8}>
                  <line x1={0} y1={4} x2={28} y2={4}
                    stroke={s.color.stroke} strokeWidth={s.dashed ? 1.5 : 2.5}
                    strokeDasharray={s.dashed ? '4,3' : undefined} />
                </svg>
                <span className="text-[10px] font-bold text-slate-600">{s.label}</span>
              </div>
            ))}
          </div>

          {/* SVG Chart */}
          <div className="w-full h-56">
            <LineChart series={allSeries} width={640} height={220} />
          </div>

          {/* Metrics grid */}
          <div className="mt-5 grid grid-cols-2 md:grid-cols-4 gap-3">
            <MetricBadge
              label="Best Segment"
              value={prediction.original.summary.best_segment}
              color="indigo"
            />
            <MetricBadge
              label="Avg CTR (Original)"
              value={`${(prediction.original.summary.avg_ctr_all * 100).toFixed(2)}%`}
              color="slate"
            />
            {hasGenerated && (
              <>
                <MetricBadge
                  label="Avg CTR (Generated)"
                  value={`${(prediction.generated.summary.avg_ctr_all * 100).toFixed(2)}%`}
                  color="green"
                />
                <MetricBadge
                  label="CTR Δ"
                  value={(() => {
                    const delta = prediction.generated.summary.avg_ctr_all - prediction.original.summary.avg_ctr_all;
                    return `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(2)}%`;
                  })()}
                  color={prediction.generated.summary.avg_ctr_all >= prediction.original.summary.avg_ctr_all ? 'green' : 'red'}
                  sub={hasGenerated ? 'vs original' : undefined}
                />
              </>
            )}
          </div>

          {/* Per-segment detail table */}
          <div className="mt-4 rounded-2xl border border-slate-100 overflow-hidden">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100">
                  <th className="px-4 py-2 text-left font-black text-slate-400 uppercase tracking-widest text-[9px]">Segment</th>
                  <th className="px-4 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Peak CTR</th>
                  <th className="px-4 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Avg CTR</th>
                  <th className="px-4 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Fatigue Day</th>
                  {hasGenerated && <th className="px-4 py-2 text-right font-black text-slate-400 uppercase tracking-widest text-[9px]">Gen Δ</th>}
                </tr>
              </thead>
              <tbody>
                {origSeries.map((seg, i) => {
                  const gen = genSeries[i];
                  const delta = gen ? gen.avg_ctr - seg.avg_ctr : null;
                  return (
                    <tr key={seg.label} className="border-b border-slate-50 hover:bg-slate-50/50 transition-colors">
                      <td className="px-4 py-2.5 font-bold text-slate-700 flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: seg.color.stroke }} />
                        {seg.country} · {seg.os}
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono font-bold text-slate-700">
                        {(seg.peak_ctr * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-2.5 text-right font-mono text-slate-500">
                        {(seg.avg_ctr * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-2.5 text-right text-slate-500">
                        {seg.fatigue_day ? `Day ${seg.fatigue_day}` : '—'}
                      </td>
                      {hasGenerated && (
                        <td className={`px-4 py-2.5 text-right font-bold font-mono ${delta >= 0 ? 'text-emerald-600' : 'text-red-500'}`}>
                          {delta !== null ? `${delta >= 0 ? '+' : ''}${(delta * 100).toFixed(2)}%` : '—'}
                        </td>
                      )}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {hasGenerated && (
            <p className="text-[9px] text-slate-400 font-bold uppercase tracking-widest mt-3 text-center">
              Dashed lines = AI Generated creative · Solid lines = Original creative
            </p>
          )}
        </div>
      )}
    </div>
  );
}
