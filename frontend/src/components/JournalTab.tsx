import { useState, useEffect, useCallback, useMemo, Fragment } from 'react'

interface Position {
  ticker: string; direction: string; entry_price: number; exit_price: number;
  stock_ret_pct: number; contribution_pct: number;
}
interface JournalDay {
  date: string; daily_return_pct: number; cumulative_pct: number; positions: Position[];
}

export default function JournalTab() {
  const [data, setData] = useState<JournalDay[]>([])
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)

  const loadJournal = useCallback(async () => {
    setLoading(true)
    try {
      const r = await fetch('/api/trade-journal')
      if (!r.ok) return
      const d = await r.json()
      setData([...d].reverse())
    } catch (e) { console.error('Error loading journal:', e) }
    finally { setLoading(false) }
  }, [])

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { loadJournal() }, [loadJournal])

  // Summary stats
  const stats = useMemo(() => {
    if (!data.length) return null
    const wins = data.filter(d => d.daily_return_pct > 0).length
    const losses = data.filter(d => d.daily_return_pct < 0).length
    const best = data.reduce((b, d) => d.daily_return_pct > b.daily_return_pct ? d : b, data[0])
    const worst = data.reduce((w, d) => d.daily_return_pct < w.daily_return_pct ? d : w, data[0])
    const avgRet = data.reduce((s, d) => s + d.daily_return_pct, 0) / data.length
    return { total: data.length, wins, losses, best, worst, avgRet }
  }, [data])

  const toggleExpand = (idx: number) => setExpandedIdx(expandedIdx === idx ? null : idx)

  return (
    <>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexShrink: 0 }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.01em' }}>Daily Rebalance Journal</div>
          <div style={{ fontSize: 12, color: 'var(--text-2)', marginTop: 3 }}>Click any row to expand position details</div>
        </div>
        <button className="btn btn-sec btn-sm" onClick={loadJournal} disabled={loading}>
          {loading ? '⟳' : '↺'} Refresh
        </button>
      </div>

      {/* Summary Stats */}
      {stats && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10, flexShrink: 0 }}>
          {[
            { label: 'Trading Days', val: stats.total, icon: '📅' },
            { label: 'Win Days', val: stats.wins, color: 'var(--green)', icon: '✅' },
            { label: 'Loss Days', val: stats.losses, color: 'var(--red)', icon: '❌' },
            { label: 'Best Day', val: `+${stats.best.daily_return_pct.toFixed(2)}%`, color: 'var(--green)', sub: stats.best.date, icon: '🏆' },
            { label: 'Worst Day', val: `${stats.worst.daily_return_pct.toFixed(2)}%`, color: 'var(--red)', sub: stats.worst.date, icon: '📉' },
          ].map(s => (
            <div key={s.label} className="card" style={{ padding: '10px 14px', display: 'flex', gap: 8, alignItems: 'center' }}>
              <span style={{ fontSize: 18, opacity: 0.7 }}>{s.icon}</span>
              <div>
                <div className="mono" style={{ fontSize: 18, fontWeight: 500, lineHeight: 1, color: s.color }}>{s.val}</div>
                <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 3 }}>
                  {s.label}{s.sub ? ` · ${s.sub}` : ''}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden' }}>
        <div style={{ overflowY: 'auto', flex: 1, minHeight: 0 }}>
          <table className="dt">
            <thead>
              <tr>
                <th style={{ width: 30 }}></th>
                <th style={{ width: 100 }}>Date</th>
                <th style={{ textAlign: 'right' }}>Daily P&L</th>
                <th style={{ textAlign: 'center', width: 100 }}>Day</th>
                <th style={{ textAlign: 'right' }}>Cum Return</th>
                <th>Positions</th>
              </tr>
            </thead>
            <tbody>
              {data.length === 0 ? (
                <tr><td colSpan={6}><div className="empty-state" style={{ padding: 30 }}>
                  <div className="empty-state-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ width: 24, height: 24, strokeWidth: 1.4 }}><path d="M4 19.5A2.5 2.5 0 016.5 17H20M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z" /></svg></div>
                  <div className="empty-state-title">No Trades Yet</div>
                  <div className="empty-state-desc">Waiting for ML rebalance to complete. Run the engine first.</div>
                </div></td></tr>
              ) : data.map((day, i) => {
                const isExpanded = expandedIdx === i
                const dColor = day.daily_return_pct > 0 ? 'var(--green)' : day.daily_return_pct < 0 ? 'var(--red)' : 'var(--text-2)'
                const cColor = day.cumulative_pct > 0 ? 'var(--green)' : day.cumulative_pct < 0 ? 'var(--red)' : 'var(--text-2)'
                const barW = Math.min(Math.abs(day.daily_return_pct) / 3 * 100, 100)
                const barColor = day.daily_return_pct > 0 ? 'var(--green)' : 'var(--red)'
                return (
                  <Fragment key={i}>
                    <tr className="expand-row" onClick={() => toggleExpand(i)} style={{ background: isExpanded ? 'rgba(59,130,246,0.04)' : undefined }}>
                      <td style={{ width: 30, textAlign: 'center' }}>
                        <span className={`expand-chevron ${isExpanded ? 'open' : ''}`}>›</span>
                      </td>
                      <td className="mono" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>{day.date}</td>
                      <td style={{ textAlign: 'right' }}>
                        <span className="mono" style={{ color: dColor, fontSize: 12, fontWeight: 500 }}>
                          {day.daily_return_pct > 0 ? '+' : ''}{day.daily_return_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4 }}>
                          <div style={{ width: 50, height: 4, background: 'var(--surf-3)', borderRadius: 2, overflow: 'hidden' }}>
                            <div style={{ width: `${barW}%`, height: '100%', background: barColor, borderRadius: 2, marginLeft: day.daily_return_pct < 0 ? 'auto' : 0 }} />
                          </div>
                        </div>
                      </td>
                      <td className="mono" style={{ textAlign: 'right', color: cColor, fontSize: 12, fontWeight: 600 }}>
                        {day.cumulative_pct > 0 ? '+' : ''}{day.cumulative_pct.toFixed(2)}%
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                          {day.positions.slice(0, 4).map((p, j) => (
                            <span key={j} className={`badge ${p.direction === 'Long' ? 'b-pos' : 'b-neg'}`} style={{ fontSize: 9 }}>
                              {p.ticker} {p.stock_ret_pct > 0 ? '+' : ''}{p.stock_ret_pct}%
                            </span>
                          ))}
                          {day.positions.length > 4 && <span style={{ fontSize: 9, color: 'var(--text-3)' }}>+{day.positions.length - 4}</span>}
                        </div>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <td colSpan={6} style={{ padding: 0, background: 'var(--surf-2)' }}>
                          <div className="expand-detail" style={{ padding: '12px 16px 12px 46px' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: 8 }}>
                              {day.positions.map((p, j) => {
                                const retColor = p.stock_ret_pct > 0 ? 'var(--green)' : p.stock_ret_pct < 0 ? 'var(--red)' : 'var(--text-2)'
                                return (
                                  <div key={j} style={{ background: 'var(--surf-1)', borderRadius: 'var(--r-sm)', padding: '8px 12px', border: '1px solid var(--border)' }}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                                      <span className="mono" style={{ fontSize: 12, fontWeight: 600 }}>{p.ticker}</span>
                                      <span className={`badge ${p.direction === 'Long' ? 'b-pos' : 'b-neg'}`} style={{ fontSize: 9 }}>{p.direction}</span>
                                    </div>
                                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-3)' }}>
                                      <span>₹{p.entry_price?.toFixed(2) || '—'} → ₹{p.exit_price?.toFixed(2) || '—'}</span>
                                      <span className="mono" style={{ color: retColor, fontWeight: 500 }}>
                                        {p.stock_ret_pct > 0 ? '+' : ''}{p.stock_ret_pct}%
                                      </span>
                                    </div>
                                    <div style={{ fontSize: 9, color: 'var(--text-3)', marginTop: 2 }}>
                                      Contribution: <span className="mono" style={{ color: p.contribution_pct > 0 ? 'var(--green)' : 'var(--red)' }}>
                                        {p.contribution_pct > 0 ? '+' : ''}{p.contribution_pct.toFixed(3)}%
                                      </span>
                                    </div>
                                  </div>
                                )
                              })}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  )
}


