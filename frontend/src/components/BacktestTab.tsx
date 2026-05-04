import { useState, useEffect, useRef } from 'react'
import { Chart, registerables } from 'chart.js'
import { colorVal, C, gridOpts, tickOpts, METRIC_TIPS } from '../utils'

Chart.register(...registerables)

interface TearsheetData {
  equity_dates: string[];
  equity_values: number[];
  bm_equity_values: number[];
  drawdown_values: number[];
  monthly: Record<string, number>;
  metrics: Record<string, number>;
  total_return: number;
  ann_return: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  win_rate: number;
  calmar: number;
  omega: number;
}

interface RegimeData {
  history: { date: string; regime: string }[];
}

export default function BacktestTab() {
  const [ts, setTs] = useState<TearsheetData | null>(null)
  const [regimes, setRegimes] = useState<RegimeData | null>(null)
  const [loading, setLoading] = useState(false)
  const equityRef = useRef<HTMLCanvasElement>(null)
  const drawdownRef = useRef<HTMLCanvasElement>(null)
  const equityChart = useRef<Chart | null>(null)
  const drawdownChart = useRef<Chart | null>(null)

  const fetchTearsheet = async () => {
    setLoading(true)
    try {
      const [res, regRes] = await Promise.all([fetch('/api/tearsheet'), fetch('/api/regime-status')])
      if (res.ok) {
        const data = await res.json()
        if (data?.sharpe != null) setTs(data)
      }
      if (regRes.ok) {
        const rData = await regRes.json()
        setRegimes(rData)
      }
    } catch (e) { console.warn('Tearsheet unavailable:', e) }
    finally { setLoading(false) }
  }

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { fetchTearsheet() }, [])

  // Init charts
  useEffect(() => {
    if (!equityRef.current || !drawdownRef.current) return
    equityChart.current = new Chart(equityRef.current, {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Portfolio', data: [], borderColor: C.blue, backgroundColor: 'rgba(59,130,246,0.08)', fill: true, tension: 0.3, borderWidth: 1.5, pointRadius: 0 },
        { label: 'NIFTY50', data: [], borderColor: 'rgba(245,158,11,0.6)', backgroundColor: 'transparent', borderDash: [5, 3], borderWidth: 1.2, pointRadius: 0, fill: false },
      ]},
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { display: true, position: 'top', align: 'end', labels: { boxWidth: 12, padding: 16, font: { size: 10, family: "'JetBrains Mono'" }, color: '#7c8aa0' } }, tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${(ctx.raw as number)?.toFixed(4)}x` } } },
        scales: {
          x: { grid: { display: false }, ticks: { ...tickOpts, maxRotation: 0 }, border: { color: 'transparent' } },
          y: { grid: gridOpts, ticks: { ...tickOpts, callback: (v: string | number) => `${parseFloat(String(v)).toFixed(2)}x` }, border: { color: 'transparent' } }
        }
      }
    })
    drawdownChart.current = new Chart(drawdownRef.current, {
      type: 'line',
      data: { labels: [], datasets: [{ label: 'Drawdown', data: [], borderColor: C.red, backgroundColor: 'rgba(239,68,68,0.08)', fill: true, tension: 0.3, borderWidth: 1.5, pointRadius: 0 }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { tooltip: { callbacks: { label: ctx => `${(ctx.raw as number)?.toFixed(2)}%` } } },
        scales: {
          x: { grid: { display: false }, ticks: { ...tickOpts, maxRotation: 0 }, border: { color: 'transparent' } },
          y: { grid: gridOpts, ticks: { ...tickOpts, callback: (v: string | number) => `${parseFloat(String(v)).toFixed(1)}%` }, border: { color: 'transparent' } }
        }
      }
    })
    return () => { equityChart.current?.destroy(); drawdownChart.current?.destroy() }
  }, [])

  useEffect(() => {
    if (!ts) return
    if (ts.equity_dates?.length && equityChart.current) {
      equityChart.current.data.labels = ts.equity_dates
      equityChart.current.data.datasets[0].data = ts.equity_values
      if (ts.bm_equity_values?.length) equityChart.current.data.datasets[1].data = ts.bm_equity_values
      equityChart.current.update()
    }
    if (ts.drawdown_values?.length && drawdownChart.current) {
      drawdownChart.current.data.labels = ts.equity_dates
      drawdownChart.current.data.datasets[0].data = ts.drawdown_values
      drawdownChart.current.update()
    }
  }, [ts])

  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

  const renderHeatmap = () => {
    if (!ts?.monthly || !Object.keys(ts.monthly).length) return <div className="empty-state" style={{ padding: 24 }}><div className="empty-state-desc">No monthly data available yet.</div></div>
    const years = [...new Set(Object.keys(ts.monthly).map((k: string) => new Date(k).getFullYear()))].sort()
    return (
      <>
        <div className="hm-grid" style={{ gridTemplateColumns: '50px repeat(12, 1fr)' }}>
          <div className="hm-cell" style={{ background: 'none' }} />
          {months.map(m => <div key={m} className="hm-cell hm-col-label">{m}</div>)}
        </div>
        {years.map(yr => (
          <div key={yr} className="hm-grid" style={{ gridTemplateColumns: '50px repeat(12, 1fr)', marginTop: 2 }}>
            <div className="hm-cell hm-row-label">{yr}</div>
            {Array.from({ length: 12 }, (_, mi) => {
              const key = Object.keys(ts.monthly).find((k: string) => { const d = new Date(k); return d.getFullYear() === yr && d.getMonth() === mi })
              const val = key ? ts.monthly[key] : null
              if (val == null) return <div key={mi} className="hm-cell" style={{ background: 'var(--surf-2)', color: 'var(--text-3)' }}>—</div>
              const intensity = Math.min(Math.abs(val) / 5, 1)
              const col = val > 0 ? `rgba(16,185,129,${0.15 + intensity * 0.65})` : `rgba(239,68,68,${0.15 + intensity * 0.65})`
              const tc = Math.abs(val) > 2 ? '#fff' : 'var(--text-2)'
              return <div key={mi} className="hm-cell" style={{ background: col, color: tc }}>{val > 0 ? '+' : ''}{val.toFixed(1)}%</div>
            })}
          </div>
        ))}
      </>
    )
  }

  const iconFor = (label: string) => {
    const map: Record<string, string> = {
      'Total Return': '📈', 'Ann. Return': '🎯', 'Sharpe Ratio': '⚖️', 'Sortino Ratio': '🛡️',
      'Max Drawdown': '📉', 'Win Rate': '🏆', 'Calmar Ratio': '🔥', 'Omega Ratio': '💎',
    }
    return map[label] || '📊'
  }

  const arrowFor = (val: number | null | undefined, threshold = 0) => {
    if (val == null) return ''
    return val > threshold ? '↑' : val < threshold ? '↓' : ''
  }

  // Calculate regime-conditioned metrics
  const regimeMetrics: Record<string, { days: number; totalReturn: number; winCount: number }> = {
    Bull: { days: 0, totalReturn: 0, winCount: 0 },
    Bear: { days: 0, totalReturn: 0, winCount: 0 },
    Sideways: { days: 0, totalReturn: 0, winCount: 0 },
  }
  
  if (ts?.equity_dates && ts?.equity_values && regimes?.history) {
    const historyMap: Record<string, string> = {}
    regimes.history.forEach((h: Record<string, string>) => historyMap[h.date] = h.regime)
    
    // Calculate daily returns
    for (let i = 1; i < ts.equity_values.length; i++) {
      const date = ts.equity_dates[i]
      const ret = (ts.equity_values[i] / ts.equity_values[i-1]) - 1
      const reg = historyMap[date]
      if (reg && regimeMetrics[reg]) {
        regimeMetrics[reg].days += 1
        regimeMetrics[reg].totalReturn += ret
        if (ret > 0) regimeMetrics[reg].winCount += 1
      }
    }
  }

  const metrics = [
    { id: 'totalReturn', label: 'Total Return', val: ts?.total_return, fmt: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`, color: colorVal(ts?.total_return) },
    { id: 'annReturn', label: 'Ann. Return', val: ts?.ann_return, fmt: (v: number) => `${v > 0 ? '+' : ''}${v.toFixed(2)}%`, color: colorVal(ts?.ann_return) },
    { id: 'sharpe', label: 'Sharpe Ratio', val: ts?.sharpe, fmt: (v: number) => v.toFixed(2), color: colorVal(ts?.sharpe) },
    { id: 'sortino', label: 'Sortino Ratio', val: ts?.sortino, fmt: (v: number) => v.toFixed(2), color: colorVal(ts?.sortino) },
    { id: 'maxDD', label: 'Max Drawdown', val: ts?.max_drawdown, fmt: (v: number) => `${v.toFixed(2)}%`, color: 'var(--red)' },
    { id: 'winRate', label: 'Win Rate', val: ts?.win_rate, fmt: (v: number) => `${v.toFixed(1)}%`, color: colorVal(ts?.win_rate, 50) },
    { id: 'calmar', label: 'Calmar Ratio', val: ts?.calmar, fmt: (v: number) => v.toFixed(2), color: colorVal(ts?.calmar) },
    { id: 'omega', label: 'Omega Ratio', val: ts?.omega, fmt: (v: number) => v.toFixed(2), color: colorVal(ts?.omega, 1) },
  ]

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.01em' }}>Performance Tear Sheet</div>
          <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 3 }}>Walk-forward backtest results · Risk-adjusted metrics</div>
        </div>
        <button className="btn btn-sec" onClick={fetchTearsheet} disabled={loading}>
          {loading ? <><span className="animate-spin" style={{ display: 'inline-block', width: 12, height: 12, border: '2px solid rgba(255,255,255,0.2)', borderTopColor: 'var(--text-1)', borderRadius: '50%' }} /> Computing...</> : '↺ Compute Tearsheet'}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
        {metrics.map(m => (
          <div key={m.id} className="tooltip-wrap an-card blue">
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
              <span style={{ fontSize: 20, lineHeight: 1, opacity: 0.7 }}>{iconFor(m.label)}</span>
              <div style={{ flex: 1 }}>
                <div className="mono" style={{ fontSize: 20, fontWeight: 500, lineHeight: 1, color: m.color }}>
                  {m.val != null ? m.fmt(m.val) : '—'}
                  {m.val != null && <span style={{ fontSize: 11, marginLeft: 4, opacity: 0.7 }}>{arrowFor(m.val, m.label === 'Max Drawdown' ? -999 : m.label === 'Win Rate' ? 50 : 0)}</span>}
                </div>
                <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 5 }}>{m.label}</div>
              </div>
            </div>
            {METRIC_TIPS[m.label] && <div className="tooltip">{METRIC_TIPS[m.label]}</div>}
          </div>
        ))}
      </div>

      <div className="card">
        <div className="card-hdr">
          <span className="card-title">Equity Curve</span>
          <span style={{ fontSize: 10, color: 'var(--text-3)' }}>Portfolio vs NIFTY50 Benchmark</span>
        </div>
        <div className="card-body"><div style={{ height: 260, position: 'relative' }}><canvas ref={equityRef} /></div></div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div className="card">
          <div className="card-hdr"><span className="card-title">Drawdown Profile</span></div>
          <div className="card-body"><div style={{ height: 180, position: 'relative' }}><canvas ref={drawdownRef} /></div></div>
        </div>
        <div className="card">
          <div className="card-hdr"><span className="card-title">Drawdown Profile</span></div>
          <div className="card-body"><div style={{ height: 180, position: 'relative' }}><canvas ref={drawdownRef} /></div></div>
        </div>
        <div className="card">
          <div className="card-hdr"><span className="card-title">Regime Performance</span><span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>HMM Classified</span></div>
          <div className="card-body">
             <table className="dt">
               <thead>
                 <tr><th>Market Regime</th><th>Days</th><th>Ann. Return</th><th>Win Rate</th></tr>
               </thead>
               <tbody>
                 {['Bull', 'Bear', 'Sideways'].map(reg => {
                    const m = regimeMetrics[reg]
                    if (!m || m.days === 0) return <tr key={reg}><td><span className="sc-sector">{reg}</span></td><td colSpan={3} className="mono" style={{ color: 'var(--text-3)' }}>No data</td></tr>
                    
                    const annRet = (m.totalReturn / m.days) * 252 * 100 // Approximation
                    const winRate = (m.winCount / m.days) * 100
                    return (
                      <tr key={reg}>
                        <td><span className="sc-sector">{reg}</span></td>
                        <td className="mono">{m.days}</td>
                        <td className="mono" style={{ color: annRet > 0 ? 'var(--green)' : 'var(--red)' }}>{annRet > 0 ? '+' : ''}{annRet.toFixed(1)}%</td>
                        <td className="mono" style={{ color: winRate > 50 ? 'var(--green)' : 'var(--red)' }}>{winRate.toFixed(1)}%</td>
                      </tr>
                    )
                 })}
               </tbody>
             </table>
          </div>
        </div>
      </div>
      
      <div className="card" style={{ marginTop: 16 }}>
        <div className="card-hdr"><span className="card-title">Monthly Returns Heatmap</span></div>
        <div style={{ padding: 16, overflowX: 'auto' }}>{renderHeatmap()}</div>
      </div>
    </>
  )
}
