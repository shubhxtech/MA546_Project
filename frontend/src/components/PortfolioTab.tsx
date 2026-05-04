import { useEffect, useRef, useState } from 'react'
import { Chart, registerables } from 'chart.js'
import type { AppState } from '../App'
import { getSector } from '../utils'

Chart.register(...registerables)

interface Props { state: AppState | null }

export default function PortfolioTab({ state }: Props) {
  const sectorRef = useRef<HTMLCanvasElement>(null)
  const convictionRef = useRef<HTMLCanvasElement>(null)
  const radarRef = useRef<HTMLCanvasElement>(null)
  const sectorChart = useRef<Chart | null>(null)
  const convictionChart = useRef<Chart | null>(null)
  const radarChart = useRef<Chart | null>(null)
  
  const [factors, setFactors] = useState<Record<string, Record<string, number>>>({})
  const [insiders, setInsiders] = useState<Array<Record<string, string | number>>>([])

  useEffect(() => {
    if (!sectorRef.current || !convictionRef.current) return
    sectorChart.current = new Chart(sectorRef.current, {
      type: 'doughnut',
      data: { labels: [], datasets: [{ data: [], backgroundColor: ['#3b82f6','#10b981','#8b5cf6','#f59e0b','#ef4444','#06b6d4','#ec4899','#64748b'], borderWidth: 0, spacing: 2, borderRadius: 3 }] },
      options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { display: true, position: 'bottom', labels: { color: '#7c8aa0', font: { size: 10 }, boxWidth: 10, padding: 10 } } } }
    })
    convictionChart.current = new Chart(convictionRef.current, {
      type: 'bar',
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 3 }] },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { maxTicksLimit: 6 }, border: { color: 'transparent' }, min: -100, max: 100 }, y: { grid: { display: false }, border: { color: 'transparent' } } }
      }
    })
    
    if (radarRef.current) {
      radarChart.current = new Chart(radarRef.current, {
        type: 'radar',
        data: { labels: ['Quality', 'Momentum', 'Low-Vol', 'Profitability', 'Value'], datasets: [{ label: 'Portfolio Exposure', data: [0,0,0,0,0], backgroundColor: 'rgba(59, 130, 246, 0.2)', borderColor: '#3b82f6', pointBackgroundColor: '#3b82f6', pointBorderColor: '#fff' }] },
        options: { responsive: true, maintainAspectRatio: false, scales: { r: { angleLines: { color: 'rgba(255,255,255,0.1)' }, grid: { color: 'rgba(255,255,255,0.1)' }, pointLabels: { color: '#7c8aa0', font: { size: 10 } }, ticks: { display: false } } }, plugins: { legend: { display: false } } }
      })
    }
    
    const fetchData = async () => {
      try {
        const [fRes, iRes] = await Promise.all([fetch('/api/factor-scores'), fetch('/api/insider-signals')])
        setFactors(await fRes.json())
        setInsiders(await iRes.json())
      } catch { /* endpoints may not be ready yet */ }
    }
    fetchData()
    // Poll slowly
    const interval = setInterval(fetchData, 10000)
    
    return () => { clearInterval(interval); sectorChart.current?.destroy(); convictionChart.current?.destroy(); radarChart.current?.destroy() }
  }, [])

  useEffect(() => {
    if (!state) return
    const w = state.recent_weights || {}
    const tickers = Object.keys(w).sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]))
    const shap = state.recent_shap || {}

    const sectors: Record<string, number> = {}
    tickers.forEach(t => { const s = getSector(t); sectors[s] = (sectors[s] || 0) + Math.abs(w[t]) })
    if (sectorChart.current) {
      sectorChart.current.data.labels = Object.keys(sectors)
      sectorChart.current.data.datasets[0].data = Object.values(sectors).map(v => +(v * 100).toFixed(1))
      sectorChart.current.update()
    }
    const convT = tickers.slice(0, 8)
    if (convictionChart.current) {
      convictionChart.current.data.labels = convT.map(t => t.replace('.NS', ''))
      convictionChart.current.data.datasets[0].data = convT.map(t => +((shap[t] || 0) * 100).toFixed(2))
      convictionChart.current.data.datasets[0].backgroundColor = convT.map(t => (shap[t] || 0) > 0 ? 'rgba(139,92,246,0.6)' : 'rgba(239,68,68,0.5)')
      convictionChart.current.update()
    }
    
    if (radarChart.current && Object.keys(factors).length > 0) {
      let q=0, m=0, v=0, p=0, y=0;
      let total_w = 0;
      tickers.forEach(t => {
        const weight = Math.abs(w[t]);
        const fac = factors[t];
        if (fac) {
          total_w += weight;
          q += fac['Quality'] * weight;
          m += fac['Momentum'] * weight;
          v += fac['Low-Vol'] * weight;
          p += fac['Profitability'] * weight;
          y += fac['Value'] * weight;
        }
      });
      if (total_w > 0) {
        radarChart.current.data.datasets[0].data = [q/total_w, m/total_w, v/total_w, p/total_w, y/total_w];
        radarChart.current.update();
      }
    }
  }, [state, factors])

  const w = state?.recent_weights || {}
  const shap = state?.recent_shap || {}
  const tickers = Object.keys(w).sort((a, b) => Math.abs(w[b]) - Math.abs(w[a]))

  // Portfolio summary KPIs
  const longExposure = tickers.reduce((s, t) => s + (w[t] > 0 ? w[t] : 0), 0) * 100
  const shortExposure = tickers.reduce((s, t) => s + (w[t] < 0 ? Math.abs(w[t]) : 0), 0) * 100
  const netExposure = longExposure - shortExposure
  const grossExposure = longExposure + shortExposure

  // Concentration warnings
  const sectors: Record<string, number> = {}
  tickers.forEach(t => { const s = getSector(t); sectors[s] = (sectors[s] || 0) + Math.abs(w[t]) * 100 })
  const warnings: string[] = []
  tickers.forEach(t => { if (Math.abs(w[t]) * 100 > 15) warnings.push(`${t.replace('.NS','')} position ${(Math.abs(w[t])*100).toFixed(1)}% exceeds 15% limit`) })
  Object.entries(sectors).forEach(([s, v]) => { if (v > 40) warnings.push(`${s} sector at ${v.toFixed(1)}% exceeds 40% cap`) })
  
  // Insider activity for portfolio holdings
  const portfolioInsiders = insiders.filter(sig => tickers.includes(`${sig.ticker}.NS`)).slice(0, 5)

  return (
    <>
      <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.01em', flexShrink: 0 }}>Live Portfolio Monitor</div>

      {/* Summary KPIs */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, flexShrink: 0 }}>
        {[
          { label: 'Long Exposure', val: `${longExposure.toFixed(1)}%`, color: 'var(--green)', icon: '📈' },
          { label: 'Short Exposure', val: `${shortExposure.toFixed(1)}%`, color: 'var(--red)', icon: '📉' },
          { label: 'Net Exposure', val: `${netExposure > 0 ? '+' : ''}${netExposure.toFixed(1)}%`, color: netExposure > 0 ? 'var(--green)' : 'var(--red)', icon: '⚖️' },
          { label: 'Positions', val: tickers.length, color: 'var(--blue)', icon: '🎯' },
        ].map(kpi => (
          <div key={kpi.label} className="card" style={{ padding: '12px 14px', display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 20, opacity: 0.7 }}>{kpi.icon}</span>
            <div>
              <div className="mono" style={{ fontSize: 20, fontWeight: 500, lineHeight: 1, color: kpi.color }}>{kpi.val}</div>
              <div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 4 }}>{kpi.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Concentration Warnings */}
      {warnings.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {warnings.map((msg, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 12px', borderRadius: 'var(--r-sm)', background: 'var(--amber-dim)', border: '1px solid rgba(245,158,11,0.25)', fontSize: 11, color: 'var(--amber)' }}>
              <span>⚠</span> {msg}
            </div>
          ))}
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: 16, flex: 1, minHeight: 0 }}>
        <div className="card" style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <div className="card-hdr" style={{ flexShrink: 0 }}>
            <span className="card-title">Active Positions</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>Gross: {grossExposure.toFixed(1)}%</span>
          </div>
          <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
            <table className="dt">
              <thead><tr><th>Ticker</th><th>Sector</th><th>Weight</th><th>Direction</th><th>Conviction</th></tr></thead>
              <tbody>
                {tickers.length ? tickers.map(t => {
                  const pct = (w[t] * 100).toFixed(2)
                  const sc = ((shap[t] || 0) * 100).toFixed(1)
                  const barW = Math.min(Math.abs(w[t]) * 100 / 20 * 100, 100)
                  const isOverweight = Math.abs(w[t]) * 100 > 15
                  return (
                    <tr key={t}>
                      <td className="mono" style={{ fontWeight: 500 }}>
                        {t.replace('.NS', '')}
                        {isOverweight && <span className="badge b-warn" style={{ marginLeft: 6, fontSize: 8, padding: '1px 4px' }}>OW</span>}
                      </td>
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
                      <td className="mono" style={{ color: 'var(--purple)' }}>{sc}%</td>
                    </tr>
                  )
                }) : (
                  <tr><td colSpan={5}><div className="empty-state" style={{ padding: 20 }}><div className="empty-state-desc">No active positions. Run the engine first.</div></div></td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div className="card">
            <div className="card-hdr"><span className="card-title">Factor Exposure</span><span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>Composite</span></div>
            <div className="card-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ height: 200, width: 200, position: 'relative' }}><canvas ref={radarRef} /></div>
            </div>
          </div>
          
          <div className="card">
            <div className="card-hdr"><span className="card-title">Sector Allocation</span></div>
            <div className="card-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ height: 180, width: 180, position: 'relative' }}><canvas ref={sectorRef} /></div>
            </div>
          </div>
          
          <div className="card">
            <div className="card-hdr"><span className="card-title">Insider Activity (Holdings)</span></div>
            <div className="card-body">
              {portfolioInsiders.length > 0 ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {portfolioInsiders.map((sig, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 11, paddingBottom: 6, borderBottom: '1px solid var(--border)' }}>
                      <div>
                        <span className="mono" style={{ color: 'var(--blue)' }}>{sig.ticker}</span>
                        <span style={{ color: 'var(--text-2)', marginLeft: 6 }}>{sig.insider_type}</span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span className={`badge ${sig.direction === 'Buy' ? 'b-pos' : 'b-neg'}`}>{sig.direction}</span>
                        <span className="mono" style={{ color: 'var(--text-1)' }}>₹{sig.value_lakhs}L</span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="empty-state" style={{ padding: 10, minHeight: 0 }}><div className="empty-state-desc">No recent insider activity in holdings.</div></div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
