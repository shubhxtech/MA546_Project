import { useState, useEffect, useMemo, useCallback } from 'react'
import { fmt } from '../utils'

interface StockData {
  ticker: string; name?: string; sector?: string; score: number;
  pe?: number; pb?: number; roe?: number; debt_equity?: number;
  net_margin?: number; rev_growth?: number; market_cap?: number; div_yield?: number;
  q_factor?: number; mom_factor?: number; vol_factor?: number; prof_factor?: number; val_factor?: number;
  composite?: number; insider_score?: number;
}

export default function ScreenerTab() {
  const [data, setData] = useState<StockData[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [sector, setSector] = useState('')
  const [sortCol, setSortCol] = useState('composite')
  const [sortAsc, setSortAsc] = useState(false)
  const [loadStatus, setLoadStatus] = useState('Loading...')
  
  const [showWeights, setShowWeights] = useState(false)
  const [weights, setWeights] = useState({ q: 0.25, m: 0.20, v: 0.20, p: 0.20, y: 0.15 })

  const handleWeightChange = (key: keyof typeof weights, val: number) => {
    setWeights(prev => ({ ...prev, [key]: val }))
  }

  // Normalize weights
  const normWeights = useMemo(() => {
    const total = weights.q + weights.m + weights.v + weights.p + weights.y
    if (total === 0) return { q: 0.2, m: 0.2, v: 0.2, p: 0.2, y: 0.2 }
    return { q: weights.q/total, m: weights.m/total, v: weights.v/total, p: weights.p/total, y: weights.y/total }
  }, [weights])

  const fetchFundamentals = useCallback(async () => {
    try {
      const res = await fetch('/api/fundamentals')
      const json = await res.json()
      
      if (!json.loaded || !json.stocks || !Object.keys(json.stocks).length) {
        setLoadStatus('Still fetching data in background (~60-90 sec left)...')
        return
      }
      
      const factorRes = await fetch('/api/factor-scores')
      const factorJson = await factorRes.json()
      
      const insiderRes = await fetch('/api/insider-signals')
      const insiderJson = await insiderRes.json()
      
      const insiderMap: Record<string, number> = {}
      if (Array.isArray(insiderJson)) {
        insiderJson.forEach(sig => {
            const t = sig.ticker + ".NS"
            insiderMap[t] = (insiderMap[t] || 0) + sig.score
        })
      }
      
      const stocks = Object.entries(json.stocks as Record<string, StockData>).map(([ticker, info]) => {
        const factors = factorJson[ticker] || {}
        return { 
          ticker, ...info,
          q_factor: factors['Quality'] || 0,
          mom_factor: factors['Momentum'] || 0,
          vol_factor: factors['Low-Vol'] || 0,
          prof_factor: factors['Profitability'] || 0,
          val_factor: factors['Value'] || 0,
          insider_score: Math.max(-1, Math.min(1, insiderMap[ticker] || 0))
        }
      }) as StockData[]

      setData(stocks)
      setLoading(false)
      setLoadStatus(`${stocks.length} stocks loaded`)
    } catch (e) {
      console.error('Fundamentals fetch error:', e)
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchFundamentals()
    const interval = setInterval(fetchFundamentals, loading ? 5000 : 30000)
    return () => clearInterval(interval)
  }, [fetchFundamentals, loading])

  const setQualityGate = async (val: number) => {
    try {
      await fetch('/api/set-fundamental-filter', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_score: val })
      })
      fetchFundamentals()
    } catch (e) { console.error('Error setting quality gate:', e) }
  }

  const handleSort = (col: string) => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
  }



  const exportCSV = () => {
    if (!data.length) return
    const headers = ['Ticker','Name','Sector','Quality Score','P/E','P/B','ROE','Debt/Equity','Net Margin','Rev Growth','Market Cap','Div Yield']
    const rows = data.map(s => [s.ticker, `"${s.name || ''}"`, `"${s.sector || ''}"`, s.score, s.pe, s.pb, s.roe, s.debt_equity, s.net_margin, s.rev_growth, s.market_cap, s.div_yield].map(v => v == null ? '' : v).join(','))
    const csv = [headers.join(','), ...rows].join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = `screener_export_${new Date().toISOString().slice(0, 10)}.csv`; a.click()
  }

  const filtered = useMemo(() => {
    const rows = data.map(s => {
      // Calculate real-time composite based on normalized slider weights
      const comp = (s.q_factor||0)*normWeights.q + (s.mom_factor||0)*normWeights.m + (s.vol_factor||0)*normWeights.v + (s.prof_factor||0)*normWeights.p + (s.val_factor||0)*normWeights.y
      return { ...s, composite: comp }
    }).filter(s => {
      if (sector && s.sector !== sector) return false
      // Filter minimum score on the new composite scale which is usually -3 to +3
      // Let's not filter out by hard score here to show all, or map minScore
      if (search && !s.ticker.toLowerCase().includes(search.toLowerCase()) && !(s.name || '').toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
    rows.sort((a: StockData, b: StockData) => {
      let va = a[sortCol as keyof StockData], vb = b[sortCol as keyof StockData]
      if (va == null) va = -999999; if (vb == null) vb = -999999
      if (typeof va === 'string' && typeof vb === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va)
      return sortAsc ? (va as number) - (vb as number) : (vb as number) - (va as number)
    })
    return rows
  }, [data, search, sector, normWeights, sortCol, sortAsc])

  const scores = filtered.map(s => s.composite || 0)
  const avg = scores.length ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2) : '0'
  const high = scores.filter(s => s >= 70).length
  const mid = scores.filter(s => s >= 40 && s < 70).length
  const low = scores.filter(s => s < 40).length
  const sectors = useMemo(() => [...new Set(data.map(s => s.sector).filter(Boolean))].sort(), [data])

  const sortArrow = (col: string) => {
    if (sortCol !== col) return null
    return <span className="sort-arrow">{sortAsc ? '▲' : '▼'}</span>
  }

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.01em' }}>Fundamental Screener</div>
          <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 3 }}>Quality scores · CAPM metrics · Live fundamentals via NSE · Data from yfinance</div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 11, color: loading ? 'var(--amber)' : 'var(--green)' }}>{loadStatus}</span>
          <button className="btn btn-sec btn-sm" onClick={fetchFundamentals}>⟳ Refresh</button>
        </div>
      </div>

      {/* KPI Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, flexShrink: 0 }}>
        {[
          { label: 'Avg Quality Score', val: avg, color: 'var(--blue)', icon: '📊' },
          { label: 'High Quality (≥70)', val: high, color: 'var(--green)', icon: '🟢' },
          { label: 'Moderate (40–69)', val: mid, color: 'var(--amber)', icon: '🟡' },
          { label: 'Low Quality (<40)', val: low, color: 'var(--red)', icon: '🔴' },
        ].map(kpi => (
          <div key={kpi.label} className="card" style={{ padding: '12px 14px', display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 20, opacity: 0.7 }}>{kpi.icon}</span>
            <div>
              <div className="mono" style={{ fontSize: 20, fontWeight: 600, color: kpi.color }}>{kpi.val}</div>
              <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 3 }}>{kpi.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Filter Bar */}
      <div className="card">
        <div className="card-body" style={{ padding: '12px 16px' }}>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <input className="ci" style={{ flex: 1, minWidth: 160 }} type="text" placeholder="Search ticker or company..." value={search} onChange={e => setSearch(e.target.value)} />
            <select className="ci" value={sector} onChange={e => setSector(e.target.value)}>
              <option value="">All Sectors</option>
              {sectors.map(s => <option key={s} value={s}>{s}</option>)}
            </select>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-2)' }}>
              <button className="btn btn-sec btn-sm" onClick={() => setShowWeights(!showWeights)}>
                {showWeights ? 'Hide Weights' : 'Factor Weights ⚙️'}
              </button>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-2)', marginLeft: 8 }}>
              <span>Min Quality:</span>
              <input type="range" min={0} max={100} defaultValue={45} style={{ width: 80 }} onMouseUp={e => setQualityGate(+e.currentTarget.value)} />
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
              <button className="btn btn-sec btn-sm" onClick={exportCSV}>⬇ CSV</button>
            </div>
          </div>
          
          {showWeights && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid var(--border)', display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 16 }}>
              {[
                {k: 'q', n: 'Quality'}, {k: 'm', n: 'Momentum'}, {k: 'v', n: 'Low-Vol'}, {k: 'p', n: 'Profitability'}, {k: 'y', n: 'Value'}
              ].map(fw => (
                <div key={fw.k} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11 }}>
                    <span style={{ color: 'var(--text-2)' }}>{fw.n}</span>
                    <span className="mono" style={{ color: 'var(--blue)' }}>{(normWeights[fw.k as keyof typeof normWeights]*100).toFixed(0)}%</span>
                  </div>
                  <input type="range" min={0} max={1} step={0.05} value={weights[fw.k as keyof typeof weights]} onChange={e => handleWeightChange(fw.k as keyof typeof weights, +e.target.value)} />
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Screener Table */}
      <div className="card" style={{ flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {loading ? (
            <div className="empty-state" style={{ padding: 48 }}>
              <div style={{ width: 24, height: 24, border: '2px solid var(--border)', borderTopColor: 'var(--blue)', borderRadius: '50%' }} className="animate-spin" />
              <div className="empty-state-title">Loading Fundamental Data</div>
              <div className="empty-state-desc">Fetching live data for all universe stocks from yfinance (~60–90 sec at startup)...</div>
            </div>
          ) : (
            <table className="dt">
              <thead>
                <tr>
                  <th onClick={() => handleSort('ticker')} className={sortCol === 'ticker' ? 'sort-active' : ''}>Ticker {sortArrow('ticker')}</th>
                  <th>Company</th>
                  <th>Sector</th>
                  <th onClick={() => handleSort('composite')} className={sortCol === 'composite' ? 'sort-active' : ''}>Composite {sortArrow('composite')}</th>
                  <th onClick={() => handleSort('q_factor')} className={sortCol === 'q_factor' ? 'sort-active' : ''}>Qual(Z) {sortArrow('q_factor')}</th>
                  <th onClick={() => handleSort('mom_factor')} className={sortCol === 'mom_factor' ? 'sort-active' : ''}>Mom(Z) {sortArrow('mom_factor')}</th>
                  <th onClick={() => handleSort('vol_factor')} className={sortCol === 'vol_factor' ? 'sort-active' : ''}>Vol(Z) {sortArrow('vol_factor')}</th>
                  <th onClick={() => handleSort('prof_factor')} className={sortCol === 'prof_factor' ? 'sort-active' : ''}>Prof(Z) {sortArrow('prof_factor')}</th>
                  <th onClick={() => handleSort('val_factor')} className={sortCol === 'val_factor' ? 'sort-active' : ''}>Val(Z) {sortArrow('val_factor')}</th>
                  <th onClick={() => handleSort('insider_score')} className={sortCol === 'insider_score' ? 'sort-active' : ''}>Insider {sortArrow('insider_score')}</th>
                  <th onClick={() => handleSort('pe')} className={sortCol === 'pe' ? 'sort-active' : ''}>P/E {sortArrow('pe')}</th>
                  <th onClick={() => handleSort('roe')} className={sortCol === 'roe' ? 'sort-active' : ''}>ROE % {sortArrow('roe')}</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr><td colSpan={12}><div className="empty-state" style={{ padding: 24 }}><div className="empty-state-desc">No stocks match your filters.</div></div></td></tr>
                ) : filtered.map(s => {
                  return (
                    <tr key={s.ticker}>
                      <td className="mono" style={{ color: 'var(--text-1)', fontWeight: 600 }}>{s.ticker.replace('.NS', '')}</td>
                      <td style={{ color: 'var(--text-2)', fontSize: 11, maxWidth: 100, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={s.name}>{s.name || '—'}</td>
                      <td><span className="sc-sector">{s.sector || '—'}</span></td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: (s.composite||0) > 1 ? 'var(--green)' : (s.composite||0) < -1 ? 'var(--red)' : 'var(--text-1)' }}>
                            {fmt(s.composite||0, 2)}
                          </span>
                        </div>
                      </td>
                      <td className={`mono ${(s.q_factor||0) > 1 ? 'green' : (s.q_factor||0) < -1 ? 'red' : ''}`}>{fmt(s.q_factor||0, 2)}</td>
                      <td className={`mono ${(s.mom_factor||0) > 1 ? 'green' : (s.mom_factor||0) < -1 ? 'red' : ''}`}>{fmt(s.mom_factor||0, 2)}</td>
                      <td className={`mono ${(s.vol_factor||0) > 1 ? 'green' : (s.vol_factor||0) < -1 ? 'red' : ''}`}>{fmt(s.vol_factor||0, 2)}</td>
                      <td className={`mono ${(s.prof_factor||0) > 1 ? 'green' : (s.prof_factor||0) < -1 ? 'red' : ''}`}>{fmt(s.prof_factor||0, 2)}</td>
                      <td className={`mono ${(s.val_factor||0) > 1 ? 'green' : (s.val_factor||0) < -1 ? 'red' : ''}`}>{fmt(s.val_factor||0, 2)}</td>
                      <td>
                        {s.insider_score !== 0 ? (
                           <span style={{ fontSize: 10, padding: '2px 6px', borderRadius: 4, background: s.insider_score! > 0 ? 'rgba(46,204,113,0.1)' : 'rgba(231,76,60,0.1)', color: s.insider_score! > 0 ? 'var(--green)' : 'var(--red)' }}>
                             {s.insider_score! > 0 ? 'BUY ' : 'SELL '}{fmt(s.insider_score, 2)}
                           </span>
                        ) : '—'}
                      </td>
                      <td className={`mono ${(s.pe || 0) > 50 || (s.pe || 0) < 0 ? 'red' : ''}`}>{fmt(s.pe)}</td>
                      <td className={`mono ${(s.roe || 0) > 0.15 ? 'green' : ''}`}>{fmt((s.roe || 0) * 100, 1, '%')}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  )
}
