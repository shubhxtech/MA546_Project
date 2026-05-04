import { useState, useEffect, useCallback, useRef } from 'react'
import Layout from './components/Layout'
import ResearchTab from './components/ResearchTab'
import SignalMatrixTab from './components/SignalMatrixTab'
import BacktestTab from './components/BacktestTab'
import PortfolioTab from './components/PortfolioTab'
import SettingsTab from './components/SettingsTab'
import JournalTab from './components/JournalTab'
import AnalyticsTab from './components/AnalyticsTab'
import ScreenerTab from './components/ScreenerTab'
import { useKeyboardShortcuts } from './utils'
import './index.css'

export type AppState = {
  status?: string
  timeline_bounds?: { min: string; max: string }
  metrics?: {
    articles_processed: number
    signals_generated: number
    nli_filtered_out: number
    final_pnl: string | null
    live_pnl: string
    turnover_cost_bps: number
  }
  logs?: Record<string, unknown>[]
  execution_logs?: string[]
  recent_weights?: Record<string, number>
  recent_shap?: Record<string, number>
  history?: {
    timestamps: string[]
    gross_allocation: number[]
    pnl_curve: number[]
    rolling_sharpe: number[]
  }
  regime?: string
}

// eslint-disable-next-line react-refresh/only-export-components
export const TABS = [
  'research', 'signals', 'backtest', 'portfolio',
  'journal', 'analytics', 'screener', 'settings'
] as const

export type TabId = (typeof TABS)[number]

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('research')
  const [state, setState] = useState<AppState | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useKeyboardShortcuts(setActiveTab)

  const fetchState = useCallback(async () => {
    try {
      const res = await fetch('/api/live-state')
      if (!res.ok) throw new Error(String(res.status))
      const data = await res.json()
      setState(data)
    } catch {
      setState(prev => prev ? { ...prev, status: 'DISCONNECTED' } : null)
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchState()
    intervalRef.current = setInterval(fetchState, 1500)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [fetchState])

  const renderTab = () => {
    switch (activeTab) {
      case 'research': return <ResearchTab state={state} />
      case 'signals': return <SignalMatrixTab />
      case 'backtest': return <BacktestTab />
      case 'portfolio': return <PortfolioTab state={state} />
      case 'settings': return <SettingsTab />
      case 'journal': return <JournalTab />
      case 'analytics': return <AnalyticsTab state={state} />
      case 'screener': return <ScreenerTab />
      default: return null
    }
  }

  return (
    <Layout activeTab={activeTab} setActiveTab={setActiveTab} state={state}>
      {renderTab()}
    </Layout>
  )
}
