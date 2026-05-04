import { useEffect, useRef, useState } from 'react'
import { Chart, registerables } from 'chart.js'
import type { AppState } from '../App'
import { getSector, C, gridOpts, tickOpts, getLogClass } from '../utils'

interface TranscriptSignal {
  ticker: string; date: string; z_score: number;
  current_tone: number; baseline_tone: number;
}

interface LogEntry {
  time: string;
  sentiment: string;
  sentiment_score: number;
  latency_ms: number;
  cached?: boolean;
  headline: string;
  nli_conf: number;
  tickers: string[];
}

const InputLabel = ({ children }: { children: string }) => (
  <div style={{ fontSize: 9, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-3)' }}>{children}</div>
)

Chart.register(...registerables)

interface Props { state: AppState | null }

const ML_MODELS = ['Linear', 'LASSO', 'Ridge', 'RF', 'GBM', 'CART', 'SVR', 'NN', 'GA']

export default function ResearchTab({ state }: Props) {
  // ── Controlled form state ──
  const [dateStart, setDateStart] = useState('')
  const [dateEnd, setDateEnd] = useState('')
  const [isWindow, setIsWindow] = useState(3)
  const [isWindowUnit, setIsWindowUnit] = useState('months')
  const [oosWindow, setOosWindow] = useState(1)
  const [oosWindowUnit, setOosWindowUnit] = useState('months')
  const [topM, setTopM] = useState(10)
  const [optProtocol, setOptProtocol] = useState('SHAP Weighting')
  const [fetchInsiderData, setFetchInsiderData] = useState(false)
  const [fetchTranscripts, setFetchTranscripts] = useState(false)
  const [mlModels, setMlModels] = useState<Record<string, boolean>>(
    Object.fromEntries(ML_MODELS.map((m, i) => [m, i < 4]))
  )
  const [isRunning, setIsRunning] = useState(false)
  
  const [leftTab, setLeftTab] = useState<'feed' | 'transcripts'>('feed')
  const [transcripts, setTranscripts] = useState<TranscriptSignal[]>([])

  const weightsRef = useRef<HTMLCanvasElement>(null)
  const shapRef = useRef<HTMLCanvasElement>(null)
  const historyRef = useRef<HTMLCanvasElement>(null)
  const pnlRef = useRef<HTMLCanvasElement>(null)
  const weightsChart = useRef<Chart | null>(null)
  const shapChart = useRef<Chart | null>(null)
  const historyChart = useRef<Chart | null>(null)
  const pnlChart = useRef<Chart | null>(null)
  const consoleRef = useRef<HTMLDivElement>(null)

  // Set dates from backend — only initializes once when bounds arrive
  const bounds = state?.timeline_bounds
  useEffect(() => {
    if (bounds) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setDateStart(prev => prev || bounds.min)
      setDateEnd(prev => prev || bounds.max)
    }
  }, [bounds])

  // Track running status
  useEffect(() => {
    const st = state?.status || ''
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsRunning(st.startsWith('CRUNCHING') || st.startsWith('GRIDSEARCH'))
  }, [state?.status])
  
  // Fetch transcript signals
  useEffect(() => {
    if (leftTab === 'transcripts') {
      fetch('/api/transcript-signals')
        .then(res => res.json())
        .then(data => setTranscripts(data))
        .catch(e => console.warn('Failed to fetch transcripts', e))
    }
  }, [leftTab, isRunning])

  // Init charts
  useEffect(() => {
    if (!weightsRef.current || !shapRef.current || !historyRef.current || !pnlRef.current) return
    Chart.defaults.color = '#7c8aa0'
    Chart.defaults.font.family = "'JetBrains Mono', monospace"
    Chart.defaults.font.size = 10
    Chart.defaults.plugins.legend.display = false
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(Chart.defaults.animation as any).duration = 300

    weightsChart.current = new Chart(weightsRef.current, {
      type: 'bar',
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 3, borderSkipped: false }] },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: { tooltip: { callbacks: { label: ctx => ` ${(ctx.raw as number).toFixed(2)}%` } } },
        scales: { x: { grid: gridOpts, ticks: tickOpts, border: { color: 'transparent' } }, y: { grid: { display: false }, border: { color: 'transparent' } } }
      }
    })
    shapChart.current = new Chart(shapRef.current, {
      type: 'bar',
      data: { labels: [], datasets: [{ data: [], backgroundColor: 'rgba(139,92,246,0.6)', borderRadius: 3, borderSkipped: false }] },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false,
        plugins: { tooltip: { callbacks: { label: ctx => ` ${((ctx.raw as number) * 100).toFixed(1)}%` } } },
        scales: { x: { grid: gridOpts, ticks: tickOpts, border: { color: 'transparent' } }, y: { grid: { display: false }, border: { color: 'transparent' } } }
      }
    })
    historyChart.current = new Chart(historyRef.current, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Gross Exposure %', data: [], borderColor: C.green, backgroundColor: 'rgba(16,185,129,0.07)', fill: true, tension: 0.4, borderWidth: 1.5, pointRadius: 0 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { x: { grid: { display: false }, ticks: { ...tickOpts, maxRotation: 0 }, border: { color: 'transparent' } }, y: { grid: gridOpts, ticks: tickOpts, min: 0, border: { color: 'transparent' } } }
      }
    })
    pnlChart.current = new Chart(pnlRef.current, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Cumulative P&L %', data: [], borderColor: C.blue, backgroundColor: 'rgba(59,130,246,0.08)', fill: true, tension: 0.3, borderWidth: 1.5, pointRadius: 0 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { tooltip: { callbacks: { label: ctx => ` ${(ctx.raw as number)?.toFixed(2)}%` } } },
        scales: { x: { grid: { display: false }, ticks: { ...tickOpts, maxRotation: 0 }, border: { color: 'transparent' } }, y: { grid: gridOpts, ticks: { ...tickOpts, callback: (v: string | number) => `${parseFloat(String(v)).toFixed(1)}%` }, border: { color: 'transparent' } } }
      }
    })
    return () => {
      weightsChart.current?.destroy(); shapChart.current?.destroy()
      historyChart.current?.destroy(); pnlChart.current?.destroy()
    }
  }, [])

  // Update charts
  useEffect(() => {
    if (!state) return
    const w = state.recent_weights || {}
    const sortedT = Object.keys(w).sort((a, b) => Math.abs(w[b]) - Math.abs(w[a])).slice(0, 12)
    const vals = sortedT.map(t => +(w[t] * 100).toFixed(2))
    if (weightsChart.current) {
      weightsChart.current.data.labels = sortedT.map(t => t.replace('.NS',''))
      weightsChart.current.data.datasets[0].data = vals
      weightsChart.current.data.datasets[0].backgroundColor = vals.map(v => v > 0 ? 'rgba(16,185,129,0.7)' : 'rgba(239,68,68,0.7)')
      weightsChart.current.update('none')
    }
    const shap = state.recent_shap || {}
    const shapT = Object.keys(shap).sort((a, b) => shap[b] - shap[a]).slice(0, 10)
    if (shapChart.current) {
      shapChart.current.data.labels = shapT.map(t => t.replace('.NS',''))
      shapChart.current.data.datasets[0].data = shapT.map(t => shap[t])
      shapChart.current.update('none')
    }
    const h = state.history || { timestamps: [], gross_allocation: [], pnl_curve: [] }
    if (historyChart.current) {
      historyChart.current.data.labels = h.timestamps
      historyChart.current.data.datasets[0].data = h.gross_allocation
      historyChart.current.update('none')
    }
    if (pnlChart.current && h.pnl_curve?.length) {
      pnlChart.current.data.labels = h.timestamps
      pnlChart.current.data.datasets[0].data = h.pnl_curve
      pnlChart.current.update('none')
    }
  }, [state])

  // Auto-scroll console
  useEffect(() => {
    if (consoleRef.current && state?.execution_logs?.length) {
      const el = consoleRef.current
      const near = el.scrollHeight - el.clientHeight <= el.scrollTop + 50
      if (near) el.scrollTop = el.scrollHeight
    }
  }, [state?.execution_logs])

  const handleStart = async () => {
    if (!dateStart || !dateEnd) { alert('Select start and end dates first.'); return }
    const active = Object.entries(mlModels).filter(([, v]) => v).map(([k]) => k)
    if (!active.length) { alert('Enable at least one ML model.'); return }
    setIsRunning(true)
    await fetch('/api/start-sim', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        start: dateStart, end: dateEnd, is_window: isWindow, is_window_unit: isWindowUnit,
        oos_window: oosWindow, oos_window_unit: oosWindowUnit, top_m: topM,
        opt_protocol: optProtocol, ml_models: active, fetch_insider_data: fetchInsiderData, fetch_transcripts: fetchTranscripts
      })
    })
  }

  const handleStop = async () => {
    await fetch('/api/stop-sim', { method: 'POST' })
  }

  const handleOptimal = async () => {
    try {
      const cfg = await (await fetch('/api/optimal-config')).json()
      if (cfg.start) setDateStart(cfg.start)
      if (cfg.end) setDateEnd(cfg.end)
      if (cfg.is_window) setIsWindow(cfg.is_window)
      if (cfg.is_window_unit) setIsWindowUnit(cfg.is_window_unit)
      if (cfg.oos_window) setOosWindow(cfg.oos_window)
      if (cfg.oos_window_unit) setOosWindowUnit(cfg.oos_window_unit)
      if (cfg.top_m) setTopM(cfg.top_m)
      if (cfg.opt_protocol) setOptProtocol(cfg.opt_protocol)
      if (cfg.ml_models?.length) {
        setMlModels(Object.fromEntries(ML_MODELS.map(m => [m, cfg.ml_models.includes(m)])))
      }
    } catch (e) { console.error(e) }
  }

  const toggleModel = (m: string) => setMlModels(prev => ({ ...prev, [m]: !prev[m] }))
  const w = state?.recent_weights || {}
  const sortedT = Object.keys(w).sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]))
  const logs = (state?.logs || []) as unknown as LogEntry[]



  return (
    <>
      {/* Control Strip */}
      <div className="card card-body" style={{ padding: '12px 16px', display: 'flex', alignItems: 'flex-end', gap: 14, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>Start Date</InputLabel>
          <input type="date" className="ci" value={dateStart} onChange={e => setDateStart(e.target.value)} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>End Date</InputLabel>
          <input type="date" className="ci" value={dateEnd} onChange={e => setDateEnd(e.target.value)} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>IS Window</InputLabel>
          <div style={{ display: 'flex', gap: 3 }}>
            <input type="number" className="ci" value={isWindow} onChange={e => setIsWindow(+e.target.value)} min={1} max={120} style={{ width: 50 }} />
            <select className="ci" value={isWindowUnit} onChange={e => setIsWindowUnit(e.target.value)}><option value="months">Mo</option><option value="days">Days</option></select>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>OOS Holding</InputLabel>
          <div style={{ display: 'flex', gap: 3 }}>
            <input type="number" className="ci" value={oosWindow} onChange={e => setOosWindow(+e.target.value)} min={1} max={120} style={{ width: 50 }} />
            <select className="ci" value={oosWindowUnit} onChange={e => setOosWindowUnit(e.target.value)}><option value="months">Mo</option><option value="days">Days</option></select>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>Top M</InputLabel>
          <input type="number" className="ci" value={topM} onChange={e => setTopM(+e.target.value)} min={1} max={50} style={{ width: 56 }} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>Optimization</InputLabel>
          <select className="ci" value={optProtocol} onChange={e => setOptProtocol(e.target.value)}>
            <option value="SHAP Weighting">SHAP Weighting</option>
            <option value="Mean-Variance">Mean-Variance (M-V)</option>
            <option value="Mean-Semivariance">Mean-Semivariance (M-SV)</option>
          </select>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>ML Models</InputLabel>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
            {ML_MODELS.map(m => (
              <div key={m} className="pill" onClick={() => toggleModel(m)}>
                <input type="checkbox" checked={mlModels[m] || false} readOnly /><span>{m}</span>
              </div>
            ))}
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <InputLabel>Data Features</InputLabel>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', alignItems: 'center' }}>
            <div className="pill" onClick={() => setFetchInsiderData(p => !p)}>
              <input type="checkbox" checked={fetchInsiderData} readOnly /><span>Insider Data</span>
            </div>
            <div className="pill" onClick={() => setFetchTranscripts(p => !p)}>
              <input type="checkbox" checked={fetchTranscripts} readOnly /><span>Earnings Transcripts</span>
            </div>
          </div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'flex-end', flexShrink: 0 }}>
          <button className="btn btn-optimal" onClick={handleOptimal}>⚡ Optimal</button>
          {isRunning ? (
            <button className="btn btn-run" onClick={handleStop} style={{ backgroundColor: 'var(--red)' }}>
              <span className="animate-spin" style={{ display: 'inline-block', width: 12, height: 12, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: '#fff', borderRadius: '50%' }} /> Stop Engine
            </button>
          ) : (
            <button className="btn btn-run" onClick={handleStart}>
              ▶ Run Engine
            </button>
          )}
        </div>
      </div>

      {/* Progress bar when running */}
      {isRunning && (
        <div className="progress-track">
          <div className="progress-fill active" style={{ width: '100%' }} />
        </div>
      )}

      {/* Metrics Strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, flexShrink: 0 }}>
        {[
          { label: 'Headlines Parsed', val: state?.metrics?.articles_processed || 0, icon: '📰' },
          { label: 'Valid Signals', val: state?.metrics?.signals_generated || 0, color: 'var(--green)', icon: '📡' },
          { label: 'Noise Filtered', val: state?.metrics?.nli_filtered_out || 0, icon: '🔇' },
          { label: 'Turnover Cost', val: `${(state?.metrics?.turnover_cost_bps || 0).toFixed(1)} bps`, color: 'var(--amber)', icon: '💸' },
        ].map(mc => (
          <div key={mc.label} className="card" style={{ padding: '12px 14px', display: 'flex', gap: 10, alignItems: 'center' }}>
            <div style={{ fontSize: 20, lineHeight: 1, opacity: 0.7 }}>{mc.icon}</div>
            <div>
              <div className="mono" style={{ fontSize: 20, fontWeight: 500, lineHeight: 1, color: mc.color }}>{typeof mc.val === 'number' ? mc.val.toLocaleString() : mc.val}</div>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 4 }}>{mc.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Main Research Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '340px 1fr', gap: 16, flex: 1, minHeight: 0 }}>
        {/* LEFT: Signal Feed + Console */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minHeight: 0, overflow: 'hidden' }}>
          <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
            <div className="card-hdr" style={{ padding: '0 12px', display: 'flex', gap: 16 }}>
              <div 
                style={{ padding: '12px 0', borderBottom: leftTab === 'feed' ? '2px solid var(--blue)' : '2px solid transparent', color: leftTab === 'feed' ? 'var(--text-1)' : 'var(--text-3)', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}
                onClick={() => setLeftTab('feed')}
              >
                Live NLP Feed
              </div>
              <div 
                style={{ padding: '12px 0', borderBottom: leftTab === 'transcripts' ? '2px solid var(--blue)' : '2px solid transparent', color: leftTab === 'transcripts' ? 'var(--text-1)' : 'var(--text-3)', cursor: 'pointer', fontSize: 12, fontWeight: 600 }}
                onClick={() => setLeftTab('transcripts')}
              >
                Earnings Transcripts
              </div>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', minHeight: 0 }}>
              {leftTab === 'feed' ? (
                logs.length === 0 ? (
                <div className="empty-state">
                  <div className="empty-state-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ width: 24, height: 24, strokeWidth: 1.4 }}><path d="M22 12 18 12 15 21 9 3 6 12 2 12" /></svg></div>
                  <div className="empty-state-title">Awaiting Engine Start</div>
                  <div className="empty-state-desc">Configure your parameters above and click "Run Engine" to begin processing news signals.</div>
                </div>
              ) : (
                logs.map((log, i) => {
                  const sc = log.sentiment === 'POSITIVE' ? 'b-pos' : log.sentiment === 'NEGATIVE' ? 'b-neg' : 'b-neu'
                  const strength = Math.min(Math.abs(log.sentiment_score || 0) / 100, 1)
                  return (
                    <div key={i} style={{ padding: '9px 14px', borderBottom: '1px solid var(--border)', transition: 'background 0.1s' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                        <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>{log.time}</span>
                        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                          {log.cached && <span style={{ fontSize: 9, color: 'var(--amber)' }} title="Cache hit — skipped LLM">⚡</span>}
                          <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>{log.latency_ms}ms</span>
                        </div>
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-1)', lineHeight: 1.4, marginBottom: 5 }}>{log.headline}</div>
                      <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap', alignItems: 'center' }}>
                        <span className={`badge ${sc}`}>{log.sentiment} {log.sentiment_score}</span>
                        <div className="spark-bar" style={{
                          width: `${strength * 32 + 4}px`,
                          background: log.sentiment === 'POSITIVE' ? 'var(--green)' : log.sentiment === 'NEGATIVE' ? 'var(--red)' : 'var(--purple)',
                          opacity: 0.5
                        }} />
                        {log.nli_conf > 0 && <span className="badge b-nli">NLI {log.nli_conf}</span>}
                        {log.tickers?.map((t: string) => <span key={t} className="badge b-ticker">{t}</span>)}
                      </div>
                    </div>
                  )
                })
                )
              ) : (
                transcripts.length === 0 ? (
                  <div className="empty-state">
                    <div className="empty-state-title">No Earnings Transcripts</div>
                    <div className="empty-state-desc">Waiting for earnings season scraper to pick up new management calls.</div>
                  </div>
                ) : (
                  transcripts.map((t, i) => (
                    <div key={i} style={{ padding: '9px 14px', borderBottom: '1px solid var(--border)' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                          <span className="badge b-ticker">{t.ticker}</span>
                          <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>{t.date}</span>
                        </div>
                        <span className={`mono ${t.z_score > 0 ? 'green' : t.z_score < 0 ? 'red' : ''}`} style={{ fontSize: 11, fontWeight: 600 }}>
                          Z: {t.z_score > 0 ? '+' : ''}{t.z_score.toFixed(2)}
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-2)', display: 'flex', justifyContent: 'space-between', marginTop: 4 }}>
                        <span>Current Tone: <span style={{ color: 'var(--text-1)' }}>{t.current_tone.toFixed(2)}</span></span>
                        <span>Rolling Baseline: <span style={{ color: 'var(--text-1)' }}>{t.baseline_tone.toFixed(2)}</span></span>
                      </div>
                    </div>
                  ))
                )
              )}
            </div>
          </div>

          {/* System Console */}
          <div className="card" style={{ height: 220, flexShrink: 0, display: 'flex', flexDirection: 'column' }}>
            <div className="card-hdr" style={{ minHeight: 36, padding: '0 14px' }}>
              <span className="card-title mono" style={{ color: 'var(--text-3)', fontSize: 11 }}>&gt; SYSTEM LOG</span>
              <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>{state?.execution_logs?.length || 0} entries</span>
            </div>
            <div ref={consoleRef} style={{ flex: 1, padding: 12, fontFamily: 'var(--font-mono)', fontSize: 10, overflowY: 'auto', lineHeight: 1.6 }}>
              {state?.execution_logs?.length ? state.execution_logs.map((msg, i) => (
                <div key={i} className={getLogClass(msg)}>{msg}</div>
              )) : (
                <span style={{ opacity: 0.5, color: 'var(--text-3)' }}>Awaiting engine sync...</span>
              )}
            </div>
          </div>
        </div>

        {/* RIGHT: Charts + Holdings */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minHeight: 0 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, flexShrink: 0 }}>
            <div className="card">
              <div className="card-hdr"><span className="card-title">ML Portfolio Weights</span></div>
              <div className="card-body"><div style={{ height: 200, position: 'relative' }}><canvas ref={weightsRef} /></div></div>
            </div>
            <div className="card">
              <div className="card-hdr"><span className="card-title">SHAP Attribution</span><span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>Feature importance</span></div>
              <div className="card-body"><div style={{ height: 200, position: 'relative' }}><canvas ref={shapRef} /></div></div>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, flexShrink: 0 }}>
            <div className="card">
              <div className="card-hdr"><span className="card-title">Cumulative P&L</span><span className="mono" style={{ fontSize: 10, color: 'var(--blue)' }}>Walk-forward</span></div>
              <div className="card-body"><div style={{ height: 160, position: 'relative' }}><canvas ref={pnlRef} /></div></div>
            </div>
            <div className="card">
              <div className="card-hdr"><span className="card-title">Gross Exposure</span><span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>∑ |W|</span></div>
              <div className="card-body"><div style={{ height: 160, position: 'relative' }}><canvas ref={historyRef} /></div></div>
            </div>
          </div>

          <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1, minHeight: 0 }}>
            <div className="card-hdr">
              <span className="card-title">Current Holdings</span>
              <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>{sortedT.length} positions</span>
            </div>
            <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
              <table className="dt">
                <thead><tr><th>Ticker</th><th>Sector</th><th>Weight</th><th>Direction</th></tr></thead>
                <tbody>
                  {sortedT.length ? sortedT.map(t => {
                    const pct = (w[t] * 100).toFixed(2)
                    const barW = Math.min(Math.abs(w[t]) * 100 / 20 * 100, 100)
                    return (
                      <tr key={t}>
                        <td className="mono" style={{ fontWeight: 500 }}>{t.replace('.NS', '')}</td>
                        <td><span className="sc-sector">{getSector(t)}</span></td>
                        <td>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span className="mono" style={{ color: w[t] > 0 ? 'var(--green)' : 'var(--red)', minWidth: 52 }}>{+pct > 0 ? '+' : ''}{pct}%</span>
                            <div style={{ width: 40, height: 4, background: 'var(--surf-3)', borderRadius: 2, overflow: 'hidden' }}>
                              <div style={{ width: `${barW}%`, height: '100%', background: w[t] > 0 ? 'var(--green)' : 'var(--red)', borderRadius: 2 }} />
                            </div>
                          </div>
                        </td>
                        <td>{w[t] > 0 ? <span className="badge b-pos">Long</span> : <span className="badge b-neg">Short</span>}</td>
                      </tr>
                    )
                  }) : (
                    <tr><td colSpan={4}><div className="empty-state" style={{ padding: 20 }}><div className="empty-state-desc">Awaiting first rebalance...</div></div></td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
