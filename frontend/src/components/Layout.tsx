import { useState, useEffect, type ReactNode } from 'react'
import type { AppState, TabId } from '../App'
import { getMarketStatus, useToasts } from '../utils'

const NAV_ITEMS: { id: TabId; label: string; icon: string }[] = [
  { id: 'research',  label: 'Research Studio', icon: 'M3 3h7v7H3zM14 3h7v7h-7zM14 14h7v7h-7zM3 14h7v7H3z' },
  { id: 'signals',   label: 'Signal Matrix',   icon: 'M9 3H5a2 2 0 00-2 2v4m6-6h10a2 2 0 012 2v4M9 3v18m0 0h10a2 2 0 002-2V9M9 21H5a2 2 0 01-2-2V9m0 0h18' },
  { id: 'backtest',  label: 'Backtest Engine',  icon: 'M22 12 18 12 15 21 9 3 6 12 2 12' },
  { id: 'portfolio', label: 'Portfolio',        icon: 'M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z' },
  { id: 'journal',   label: 'Trade Journal',    icon: 'M4 19.5A2.5 2.5 0 0 1 6.5 17H20M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z' },
  { id: 'analytics', label: 'Analytics',        icon: 'M16 8v8m-4-5v5m-4-2v2m-2 4h16a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z' },
  { id: 'screener',  label: 'Screener',         icon: 'M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z' },
]

interface Props {
  activeTab: TabId
  setActiveTab: (t: TabId) => void
  state: AppState | null
  children: ReactNode
}

export default function Layout({ activeTab, setActiveTab, state, children }: Props) {
  const [expanded, setExpanded] = useState(false)
  const [clock, setClock] = useState('')
  const { toasts, remove } = useToasts()

  const m = state?.metrics
  const st = state?.status || 'CONNECTING...'
  const displayPnl = m?.final_pnl || m?.live_pnl
  const isPos = displayPnl?.startsWith('+')
  const isNeg = displayPnl?.startsWith('-')
  const market = getMarketStatus()

  const dotClass = st.startsWith('CRUNCHING') || st.startsWith('GRIDSEARCH')
    ? 'crunching' : st === 'DISCONNECTED' ? 'off' : 'live'

  // Live clock
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setClock(now.toLocaleTimeString('en-IN', { timeZone: 'Asia/Kolkata', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }))
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <div style={{ display: 'grid', gridTemplateRows: '48px 1fr', gridTemplateColumns: 'auto 1fr', height: '100vh' }}>
      {/* ── TOPBAR ── */}
      <header style={{
        gridColumn: '1/-1', display: 'flex', alignItems: 'center', padding: '0 16px',
        background: 'var(--surf-1)', borderBottom: '1px solid var(--border)', gap: 14, zIndex: 200
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 9, flexShrink: 0 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 8,
            background: 'linear-gradient(135deg,#3b82f6,#8b5cf6)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 14, fontWeight: 700, color: '#fff', letterSpacing: '-0.02em'
          }}>IQ</div>
          <div style={{ lineHeight: 1.2 }}>
            <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)', letterSpacing: '-0.01em' }}>India NLP Quant</div>
            <div style={{ fontSize: 9, color: 'var(--text-3)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Research Terminal · NSE/BSE</div>
          </div>
        </div>

        <div style={{ width: 1, height: 22, background: 'var(--border)', flexShrink: 0 }} />

        {/* Status pill */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 6, padding: '4px 10px',
          borderRadius: 20, background: 'var(--surf-2)', border: '1px solid var(--border)',
          fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-2)', whiteSpace: 'nowrap', flexShrink: 0
        }}>
          <div style={{
            width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
            background: dotClass === 'live' ? 'var(--green)' : dotClass === 'crunching' ? 'var(--amber)' : 'var(--text-3)',
            boxShadow: dotClass === 'live' ? '0 0 6px var(--green)' : dotClass === 'crunching' ? '0 0 6px var(--amber)' : 'none',
            animation: dotClass !== 'off' ? `blink ${dotClass === 'crunching' ? '0.8s' : '2s'} ease-in-out infinite` : 'none'
          }} />
          <span>{st}</span>
        </div>
        
        {/* Regime pill */}
        {state?.regime && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: 4, padding: '4px 8px',
            borderRadius: 6, background: state.regime === 'Bull' ? 'rgba(16, 185, 129, 0.1)' : state.regime === 'Bear' ? 'rgba(239, 68, 68, 0.1)' : 'var(--surf-2)', 
            border: `1px solid ${state.regime === 'Bull' ? 'rgba(16, 185, 129, 0.3)' : state.regime === 'Bear' ? 'rgba(239, 68, 68, 0.3)' : 'var(--border)'}`,
            fontFamily: 'var(--font-mono)', fontSize: 10, color: state.regime === 'Bull' ? 'var(--green)' : state.regime === 'Bear' ? 'var(--red)' : 'var(--text-2)', flexShrink: 0
          }}>
            <span>REGIME: {state.regime.toUpperCase()}</span>
          </div>
        )}

        {/* Market status + Clock */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0 }}>
          <span className={`mono ${market.class}`} style={{ fontSize: 10 }}>{market.label}</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--text-3)' }}>IST {clock}</span>
        </div>

        <div style={{ width: 1, height: 22, background: 'var(--border)', flexShrink: 0 }} />

        {/* Top metrics */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 20, marginLeft: 'auto' }}>
          {[
            { val: m?.articles_processed, label: 'Parsed', color: undefined },
            { val: m?.signals_generated, label: 'Signals', color: 'var(--green)' },
            { val: m?.nli_filtered_out, label: 'Filtered', color: undefined },
          ].map(tm => (
            <div key={tm.label} style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', padding: '2px 8px', borderRadius: 6, cursor: 'default' }}>
              <div className="mono" style={{ fontSize: 13, fontWeight: 500, lineHeight: 1, color: tm.color || 'var(--text-2)' }}>{(tm.val || 0).toLocaleString()}</div>
              <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.08em', color: 'var(--text-3)', marginTop: 2 }}>{tm.label}</div>
            </div>
          ))}
        </div>

        {/* PnL Badge */}
        {displayPnl && displayPnl !== '0.00%' && (
          <div className={`pnl-badge ${isPos ? 'pnl-pos' : isNeg ? 'pnl-neg' : ''}`} style={{
            padding: '5px 14px', borderRadius: 'var(--r-sm)', fontFamily: 'var(--font-mono)',
            fontSize: 14, fontWeight: 600, border: '1px solid',
            background: isPos ? 'var(--green-dim)' : isNeg ? 'var(--red-dim)' : 'var(--surf-2)',
            color: isPos ? 'var(--green)' : isNeg ? 'var(--red)' : 'var(--text-1)',
            borderColor: isPos ? 'var(--green)' : isNeg ? 'var(--red)' : 'var(--border)',
            flexShrink: 0,
          }}>
            {m?.final_pnl ? 'Final' : 'Live'} P&L: {displayPnl}
          </div>
        )}
      </header>

      {/* ── SIDEBAR ── */}
      <nav className={`sidebar ${expanded ? 'expanded' : ''}`}>
        {NAV_ITEMS.map((item, idx) => (
          <button
            key={item.id}
            onClick={() => setActiveTab(item.id)}
            className={`nav-btn ${activeTab === item.id ? 'active' : ''}`}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ width: 18, height: 18, strokeWidth: 1.6, flexShrink: 0 }}>
              <path d={item.icon} />
            </svg>
            <span className="nav-label">{item.label}</span>
            <span className="nav-shortcut">{idx + 1}</span>
          </button>
        ))}

        <div className="nav-divider" />

        {/* Settings */}
        <button
          onClick={() => setActiveTab('settings')}
          className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
        >
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ width: 18, height: 18, strokeWidth: 1.6, flexShrink: 0 }}>
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
          <span className="nav-label">Settings</span>
          <span className="nav-shortcut">8</span>
        </button>

        {/* Expand/Collapse */}
        <button className="sidebar-toggle" onClick={() => setExpanded(!expanded)} title={expanded ? 'Collapse' : 'Expand'}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" style={{ width: 16, height: 16, strokeWidth: 1.6, transition: 'transform 0.25s', transform: expanded ? 'rotate(180deg)' : 'none' }}>
            <path d="M9 18l6-6-6-6" />
          </svg>
        </button>
      </nav>

      {/* ── CONTENT AREA ── */}
      <main style={{ overflow: 'hidden', position: 'relative' }}>
        <div className="tab-content" key={activeTab} style={{ height: '100%', overflowY: 'auto', padding: 18, display: 'flex', flexDirection: 'column', gap: 16 }}>
          {children}
        </div>
      </main>

      {/* ── TOASTS ── */}
      {toasts.length > 0 && (
        <div className="toast-container">
          {toasts.map(t => (
            <div key={t.id} className={`toast toast-${t.type}`} onClick={() => remove(t.id)}>
              {t.type === 'success' ? '✓' : t.type === 'error' ? '✕' : 'ℹ'} {t.message}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
