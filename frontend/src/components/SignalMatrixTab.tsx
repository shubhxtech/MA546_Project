import { useState, useEffect, useCallback, useMemo } from 'react'

interface Signal { date: string; tickers: string[]; score: number; nli: number; headline: string; source?: string }

const PAGE_SIZE = 50

export default function SignalMatrixTab() {
  const [cache, setCache] = useState<Signal[]>([])
  const [search, setSearch] = useState('')
  const [sortVal, setSortVal] = useState('time_desc')
  const [tickerFilter, setTickerFilter] = useState('')
  const [sourceFilter, setSourceFilter] = useState('')
  const [sortCol, setSortCol] = useState<string | null>(null)
  const [sortAsc, setSortAsc] = useState(true)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(false)

  const fetchDB = useCallback(async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/signals-db')
      const data = await res.json()
      setCache(data)
    } catch (e) { console.error('DB fetch error:', e) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => {
    let isMounted = true
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (isMounted) fetchDB()
    return () => { isMounted = false }
  }, [fetchDB])

  const tickers = useMemo(() => [...new Set(cache.flatMap(r => r.tickers))].sort(), [cache])

  const handleSort = (col: string) => {
    if (sortCol === col) setSortAsc(!sortAsc)
    else { setSortCol(col); setSortAsc(false) }
    setPage(1)
  }

  const sortArrow = (col: string) => {
    if (sortCol !== col) return null
    return <span className="sort-arrow">{sortAsc ? '▲' : '▼'}</span>
  }

  const filteredRows = useMemo(() => {
    let rows = cache.filter(r => {
      const matchQ = r.headline.toLowerCase().includes(search.toLowerCase()) || r.tickers.join(',').toLowerCase().includes(search.toLowerCase())
      const matchT = !tickerFilter || r.tickers.includes(tickerFilter)
      const matchS = !sourceFilter || r.source === sourceFilter || (sourceFilter === 'News' && r.source !== 'Transcript')
      return matchQ && matchT && matchS
    })

    if (sortCol) {
      rows = [...rows].sort((a, b) => {
        const av = a[sortCol as keyof typeof a] as string | number
        const bv = b[sortCol as keyof typeof b] as string | number
        return sortAsc ? (av > bv ? 1 : -1) : (av < bv ? 1 : -1)
      })
    } else {
      rows = [...rows]
      if (sortVal === 'score_desc') rows.sort((a, b) => b.score - a.score)
      else if (sortVal === 'score_asc') rows.sort((a, b) => a.score - b.score)
      else if (sortVal === 'abs_desc') rows.sort((a, b) => Math.abs(b.score) - Math.abs(a.score))
      else if (sortVal === 'time_desc') rows.reverse()
    }
    return rows
  }, [cache, search, tickerFilter, sourceFilter, sortCol, sortAsc, sortVal])

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE))
  const pageRows = filteredRows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)

  // Stats
  const avgScore = filteredRows.length ? (filteredRows.reduce((s, r) => s + r.score, 0) / filteredRows.length) : 0
  const posCount = filteredRows.filter(r => r.score > 10).length
  const negCount = filteredRows.filter(r => r.score < -10).length

  const exportCSV = () => {
    if (!cache.length) return
    const csv = ['Date,Tickers,Score,NLI Confidence,Headline', ...cache.map(r => `"${r.date}","${r.tickers.join(';')}",${r.score},${r.nli},"${r.headline.replace(/"/g, "'")}"`)].join('\n')
    const a = document.createElement('a')
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }))
    a.download = 'signals_db.csv'; a.click()
  }

  return (
    <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
      <div className="card-hdr" style={{ flexShrink: 0 }}>
        <span className="card-title">Central Signal Database</span>
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: 8, fontSize: 10 }}>
            <span className="mono" style={{ color: 'var(--green)' }}>▲ {posCount}</span>
            <span className="mono" style={{ color: 'var(--red)' }}>▼ {negCount}</span>
            <span className="mono" style={{ color: 'var(--text-3)' }}>μ {avgScore.toFixed(1)}</span>
          </div>
          <button className="btn btn-sec btn-sm" onClick={fetchDB} disabled={loading}>
            {loading ? '⟳' : '↺'} Refresh
          </button>
        </div>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', padding: '10px 16px', borderBottom: '1px solid var(--border)', background: 'var(--surf-2)', flexWrap: 'wrap', flexShrink: 0 }}>
        <input type="text" className="ci" style={{ flex: 1, minWidth: 160 }} placeholder="Search ticker or headline..." value={search} onChange={e => { setSearch(e.target.value); setPage(1) }} />
        <select className="ci" value={sortVal} onChange={e => { setSortVal(e.target.value); setSortCol(null); setPage(1) }}>
          <option value="time_desc">Newest First</option>
          <option value="time_asc">Oldest First</option>
          <option value="score_desc">Highest Score</option>
          <option value="score_asc">Lowest Score</option>
          <option value="abs_desc">Strongest Signal</option>
        </select>
        <select className="ci" value={sourceFilter} onChange={e => { setSourceFilter(e.target.value); setPage(1) }}>
          <option value="">All Sources</option>
          <option value="News">News Headlines</option>
          <option value="Transcript">Earnings Calls</option>
        </select>
        <select className="ci" value={tickerFilter} onChange={e => { setTickerFilter(e.target.value); setPage(1) }}>
          <option value="">All Tickers</option>
          {tickers.map(t => <option key={t}>{t}</option>)}
        </select>
        <button className="btn btn-sec btn-sm" onClick={exportCSV}>⬇ CSV</button>
      </div>

      {/* Table */}
      <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
        <table className="dt">
          <thead>
            <tr>
              <th onClick={() => handleSort('date')} className={sortCol === 'date' ? 'sort-active' : ''}>Date {sortArrow('date')}</th>
              <th onClick={() => handleSort('source')} className={sortCol === 'source' ? 'sort-active' : ''}>Source {sortArrow('source')}</th>
              <th onClick={() => handleSort('tickers')} className={sortCol === 'tickers' ? 'sort-active' : ''}>Tickers {sortArrow('tickers')}</th>
              <th onClick={() => handleSort('score')} className={sortCol === 'score' ? 'sort-active' : ''}>Score {sortArrow('score')}</th>
              <th onClick={() => handleSort('nli')} className={sortCol === 'nli' ? 'sort-active' : ''}>NLI {sortArrow('nli')}</th>
              <th>Headline</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr><td colSpan={5}><div className="empty-state" style={{ padding: 24 }}><div className="empty-state-desc">No matching signals found.</div></div></td></tr>
            ) : pageRows.map((r, i) => {
              const tc = r.score > 10 ? 'var(--green)' : r.score < -10 ? 'var(--red)' : 'var(--text-2)'
              const strength = Math.min(Math.abs(r.score) / 80, 1)
              const bgAlpha = strength * 0.04
              const bg = r.score > 10 ? `rgba(16,185,129,${bgAlpha})` : r.score < -10 ? `rgba(239,68,68,${bgAlpha})` : 'transparent'
              return (
                <tr key={i} style={{ background: bg }}>
                  <td className="mono" style={{ color: 'var(--text-3)', fontSize: 10, whiteSpace: 'nowrap' }}>{r.date}</td>
                  <td className="mono" style={{ color: r.source === 'Transcript' ? 'var(--blue)' : 'var(--text-3)', fontSize: 10 }}>{r.source === 'Transcript' ? 'Earnings' : 'News'}</td>
                  <td className="mono">{r.tickers.map(t => <span key={t} className="badge b-ticker" style={{ marginRight: 3 }}>{t}</span>)}</td>
                  <td>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span className="mono" style={{ color: tc, fontWeight: 500 }}>{(r.score > 0 ? '+' : '') + r.score.toFixed(2)}</span>
                      <div className="score-bar-track">
                        <div className="score-bar-fill" style={{ width: `${strength * 100}%`, background: r.score > 0 ? 'var(--green)' : 'var(--red)' }} />
                      </div>
                    </div>
                  </td>
                  <td className="mono" style={{ color: 'var(--text-2)' }}>{r.nli.toFixed(3)}</td>
                  <td style={{ fontSize: 12, color: 'var(--text-1)', maxWidth: 500 }}>{r.headline}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 16px', borderTop: '1px solid var(--border)', background: 'var(--surf-2)', flexShrink: 0 }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>{filteredRows.length.toLocaleString()} signals · Page {page} of {totalPages}</span>
        <div className="pagination">
          <button className="page-btn" disabled={page <= 1} onClick={() => setPage(1)}>«</button>
          <button className="page-btn" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>‹</button>
          {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
            const p = Math.max(1, Math.min(page - 2, totalPages - 4)) + i
            if (p > totalPages) return null
            return <button key={p} className={`page-btn ${p === page ? 'active' : ''}`} onClick={() => setPage(p)}>{p}</button>
          })}
          <button className="page-btn" disabled={page >= totalPages} onClick={() => setPage(p => p + 1)}>›</button>
          <button className="page-btn" disabled={page >= totalPages} onClick={() => setPage(totalPages)}>»</button>
        </div>
      </div>
    </div>
  )
}
