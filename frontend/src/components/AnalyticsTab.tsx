import { useState, useEffect, useRef } from 'react'
import { Chart, registerables } from 'chart.js'
import { colorVal, C, gridOpts, tickOpts, METRIC_TIPS } from '../utils'

import type { AppState } from '../App'

Chart.register(...registerables)

interface TearsheetData {
  total_return: number;
  bm_total_return: number;
  sharpe: number;
  max_drawdown: number;
  beta: number;
  alpha_ann: number;
  info_ratio: number;
  tracking_error: number;
  calmar: number;
  omega: number;
  turnover_cost_bps: number;
  rolling_sharpe: number[];
}

interface Props { state: AppState | null }

export default function AnalyticsTab({ state }: Props) {
  const [ts, setTs] = useState<TearsheetData | null>(null)
  const [loading, setLoading] = useState(false)
  const [factors, setFactors] = useState<Record<string, Record<string, number>>>({})
  
  const rollingRef = useRef<HTMLCanvasElement>(null)
  const rollingChart = useRef<Chart | null>(null)
  
  const factorRef = useRef<HTMLCanvasElement>(null)
  const factorChart = useRef<Chart | null>(null)

  const fetchAnalytics = async () => {
    setLoading(true)
    try {
      const [res, facRes] = await Promise.all([fetch('/api/tearsheet'), fetch('/api/factor-scores')])
      if (res.ok) {
        const data = await res.json()
        if (data && Object.keys(data).length) setTs(data)
      }
      if (facRes.ok) {
        setFactors(await facRes.json())
      }
    } catch (e) { console.warn('Analytics fetch error:', e) }
    finally { setLoading(false) }
  }

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { fetchAnalytics() }, [])

  useEffect(() => {
    if (!rollingRef.current) return
    rollingChart.current = new Chart(rollingRef.current, {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Rolling Sharpe', data: [], borderColor: C.purple, backgroundColor: 'rgba(139,92,246,0.07)', fill: true, tension: 0.4, borderWidth: 1.5, pointRadius: 0 },
        { label: 'Sharpe=0', data: [], borderColor: 'rgba(255,255,255,0.1)', borderDash: [4, 4], borderWidth: 1, pointRadius: 0, fill: false },
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 }, border: { color: 'transparent' } },
          y: { grid: gridOpts, ticks: { ...tickOpts, callback: (v: string | number) => `${parseFloat(String(v)).toFixed(1)}` }, border: { color: 'transparent' } }
        }
      }
    })
    
    if (factorRef.current) {
      factorChart.current = new Chart(factorRef.current, {
        type: 'bar',
        data: { labels: ['Quality', 'Momentum', 'Low-Vol', 'Profitability', 'Value'], datasets: [
          { label: 'Longs', data: [0,0,0,0,0], backgroundColor: 'rgba(16, 185, 129, 0.8)', borderRadius: 3 },
          { label: 'Shorts', data: [0,0,0,0,0], backgroundColor: 'rgba(239, 68, 68, 0.8)', borderRadius: 3 },
        ]},
        options: {
          indexAxis: 'y', responsive: true, maintainAspectRatio: false,
          plugins: { legend: { display: true, position: 'bottom', labels: { color: '#7c8aa0', font: { size: 10 } } } },
          scales: {
            x: { stacked: false, grid: { color: 'rgba(255,255,255,0.04)' }, border: { color: 'transparent' } },
            y: { stacked: false, grid: { display: false }, border: { color: 'transparent' }, ticks: { color: '#7c8aa0' } }
          }
        }
      })
    }
    
    return () => { rollingChart.current?.destroy(); factorChart.current?.destroy() }
  }, [])

  useEffect(() => {
    if (!ts?.rolling_sharpe?.length || !rollingChart.current) return
    const labels = ts.rolling_sharpe.map((_: unknown, i: number) => i % 20 === 0 ? `Day ${i + 21}` : '')
    rollingChart.current.data.labels = labels
    rollingChart.current.data.datasets[0].data = ts.rolling_sharpe
    rollingChart.current.data.datasets[1].data = ts.rolling_sharpe.map(() => 0)
    rollingChart.current.update()
  }, [ts])
  
  useEffect(() => {
    if (!state?.recent_weights || !factorChart.current || !Object.keys(factors).length) return
    const w = state.recent_weights
    const fLabels = ['Quality', 'Momentum', 'Low-Vol', 'Profitability', 'Value']
    
    const long_exposures = [0,0,0,0,0]
    const short_exposures = [0,0,0,0,0]
    let long_w = 0, short_w = 0
    
    Object.keys(w).forEach(t => {
      const weight = w[t]
      const fac = factors[t]
      if (!fac) return
      if (weight > 0) {
        long_w += weight
        fLabels.forEach((label, i) => long_exposures[i] += fac[label] * weight)
      } else if (weight < 0) {
        short_w += Math.abs(weight)
        fLabels.forEach((label, i) => short_exposures[i] += fac[label] * Math.abs(weight))
      }
    })
    
    if (long_w > 0) fLabels.forEach((_, i) => long_exposures[i] /= long_w)
    if (short_w > 0) fLabels.forEach((_, i) => short_exposures[i] /= short_w)
      
    factorChart.current.data.datasets[0].data = long_exposures
    factorChart.current.data.datasets[1].data = short_exposures
    factorChart.current.update()
  }, [state, factors])

  const iconFor = (label: string) => {
    const map: Record<string, string> = {
      'Information Ratio': '📊', 'CAPM Alpha (Ann.)': '🎯', 'Beta': '📐', 'Tracking Error': '📏',
      'Calmar Ratio': '🔥', 'Omega Ratio': '💎', 'Turnover Cost': '💸', 'Benchmark Return': '🏛️',
    }
    return map[label] || '📊'
  }

  const metrics = [
    { id: 'anIR', label: 'Information Ratio', val: ts?.info_ratio, fmt: (v: number) => v.toFixed(3), color: colorVal(ts?.info_ratio), accent: 'blue', sub: 'Active return / tracking error' },
    { id: 'anAlpha', label: 'CAPM Alpha (Ann.)', val: ts?.alpha_ann, fmt: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`, color: colorVal(ts?.alpha_ann), accent: 'purple', sub: 'vs NIFTY50 · 6.5% risk-free' },
    { id: 'anBeta', label: 'Beta', val: ts?.beta, fmt: (v: number) => v.toFixed(3), color: ts?.beta && ts.beta > 1 ? 'var(--amber)' : 'var(--green)', accent: 'amber', sub: 'Market sensitivity' },
    { id: 'anTE', label: 'Tracking Error', val: ts?.tracking_error, fmt: (v: number) => `${v.toFixed(2)}%`, color: 'var(--cyan)', accent: 'cyan', sub: 'Annualised active risk' },
    { id: 'anCalmar', label: 'Calmar Ratio', val: ts?.calmar, fmt: (v: number) => v.toFixed(2), color: colorVal(ts?.calmar), accent: 'green', sub: 'Ann. return / max drawdown' },
    { id: 'anOmega', label: 'Omega Ratio', val: ts?.omega, fmt: (v: number) => v.toFixed(2), color: colorVal(ts?.omega, 1), accent: 'green', sub: 'Gains / losses above threshold' },
    { id: 'anTurnover', label: 'Turnover Cost', val: ts?.turnover_cost_bps, fmt: (v: number) => `${v.toFixed(1)} bps`, color: 'var(--amber)', accent: 'red', sub: 'Cumulative cost (5bps/unit)' },
    { id: 'anBMRet', label: 'Benchmark Return', val: ts?.bm_total_return, fmt: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`, color: colorVal(ts?.bm_total_return), accent: 'blue', sub: 'NIFTY50 total return' },
  ]

  const bmRows = ts?.total_return != null && ts?.bm_total_return != null ? [
    ['Total Return', `${ts.total_return > 0 ? '+' : ''}${ts.total_return.toFixed(2)}%`, `${ts.bm_total_return > 0 ? '+' : ''}${ts.bm_total_return.toFixed(2)}%`, `${(ts.total_return - ts.bm_total_return).toFixed(2)}%`],
    ['Sharpe Ratio', ts.sharpe?.toFixed(3) ?? '—', '—', '—'],
    ['Max Drawdown', `${ts.max_drawdown?.toFixed(2)}%`, '—', '—'],
    ['Beta to NIFTY', ts.beta?.toFixed(3) ?? '—', '1.000', '—'],
    ['CAPM Alpha', ts.alpha_ann != null ? `${ts.alpha_ann > 0 ? '+' : ''}${ts.alpha_ann.toFixed(2)}%` : '—', '0.00%', ts.alpha_ann != null ? `${ts.alpha_ann.toFixed(2)}%` : '—'],
    ['Info Ratio', ts.info_ratio?.toFixed(3) ?? '—', '—', '—'],
    ['Tracking Error', ts.tracking_error != null ? `${ts.tracking_error.toFixed(2)}%` : '—', '0.00%', '—'],
  ] : []

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.01em' }}>Advanced Analytics</div>
          <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 3 }}>Benchmark-relative risk metrics · CAPM attribution · Rolling factor analysis</div>
        </div>
        <button className="btn btn-sec" onClick={fetchAnalytics} disabled={loading}>
          {loading ? <><span className="animate-spin" style={{ display: 'inline-block', width: 12, height: 12, border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--text-1)', borderRadius: '50%' }} /> Computing...</> : '↻ Refresh'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12 }}>
        {metrics.map(m => (
          <div key={m.id} className={`tooltip-wrap an-card ${m.accent}`}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <span style={{ fontSize: 20, lineHeight: 1, opacity: 0.7 }}>{iconFor(m.label)}</span>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 10, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{m.label}</div>
                <div className="mono" style={{ fontSize: 22, fontWeight: 600, lineHeight: 1, marginTop: 4, color: m.color }}>{m.val != null ? m.fmt(m.val) : '—'}</div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--text-2)', marginTop: 6 }}>{m.sub}</div>
              </div>
            </div>
            {METRIC_TIPS[m.label] && <div className="tooltip">{METRIC_TIPS[m.label]}</div>}
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-hdr">
          <span className="card-title">Rolling 21-Day Sharpe Ratio</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>Annualised · 21-day window</span>
        </div>
        <div className="card-body"><div style={{ height: 220, position: 'relative' }}><canvas ref={rollingRef} /></div></div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div className="card">
          <div className="card-hdr">
            <span className="card-title">Strategy vs NIFTY50 Attribution</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>Side-by-side comparison</span>
          </div>
          <div className="card-body">
            <table className="dt">
              <thead><tr><th>Metric</th><th>Strategy</th><th>NIFTY50</th><th>Excess</th></tr></thead>
              <tbody>
                {bmRows.length ? bmRows.map((r, i) => {
                  const excessColor = r[3] !== '—' ? (parseFloat(r[3]) > 0 ? 'var(--green)' : 'var(--red)') : 'var(--text-2)'
                  return (
                    <tr key={i} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.015)' }}>
                      <td style={{ color: 'var(--text-2)', fontWeight: 500 }}>{r[0]}</td>
                      <td className="mono" style={{ color: 'var(--text-1)' }}>{r[1]}</td>
                      <td className="mono" style={{ color: 'var(--text-3)' }}>{r[2]}</td>
                      <td className="mono" style={{ color: excessColor, fontWeight: 600 }}>{r[3]}</td>
                    </tr>
                  )
                }) : (
                  <tr><td colSpan={4}><div className="empty-state" style={{ padding: 24 }}><div className="empty-state-desc">Run backtest to compute benchmark attribution.</div></div></td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
        
        <div className="card">
          <div className="card-hdr">
            <span className="card-title">Factor Attribution</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>Long vs Short Leg (Z-Scores)</span>
          </div>
          <div className="card-body"><div style={{ height: 220, position: 'relative' }}><canvas ref={factorRef} /></div></div>
        </div>
      </div>
    </>
  )
}
