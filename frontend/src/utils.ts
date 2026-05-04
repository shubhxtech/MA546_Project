import { useEffect, useCallback, useState } from 'react'
import type { TabId } from './App'

// ── Sector mapping ──
export const SECTOR_MAP: Record<string, string> = {
  'HDFCBANK.NS':'Banking','ICICIBANK.NS':'Banking','SBIN.NS':'Banking',
  'KOTAKBANK.NS':'Banking','AXISBANK.NS':'Banking','BANKNIFTY.NS':'Banking',
  'INDUSINDBK.NS':'Banking','BANDHANBNK.NS':'Banking','FEDERALBNK.NS':'Banking',
  'TCS.NS':'IT','INFY.NS':'IT','WIPRO.NS':'IT','HCLTECH.NS':'IT','TECHM.NS':'IT',
  'LTI.NS':'IT','MPHASIS.NS':'IT','PERSISTENT.NS':'IT','COFORGE.NS':'IT',
  'RELIANCE.NS':'Energy','ONGC.NS':'Energy','COALINDIA.NS':'Energy','NTPC.NS':'Energy',
  'POWERGRID.NS':'Energy','TATAPOWER.NS':'Energy','ADANIGREEN.NS':'Energy',
  'TATAMOTORS.NS':'Auto','MARUTI.NS':'Auto','M&M.NS':'Auto','BAJAJ-AUTO.NS':'Auto',
  'HEROMOTOCO.NS':'Auto','EICHERMOT.NS':'Auto','TVSMOTORCO.NS':'Auto',
  'SUNPHARMA.NS':'Pharma','DRREDDY.NS':'Pharma','CIPLA.NS':'Pharma',
  'DIVISLAB.NS':'Pharma','BIOCON.NS':'Pharma','LUPIN.NS':'Pharma',
  'NESTLEIND.NS':'FMCG','HINDUNILVR.NS':'FMCG','ITC.NS':'FMCG',
  'BRITANNIA.NS':'FMCG','DABUR.NS':'FMCG','MARICO.NS':'FMCG',
  'ULTRACEMCO.NS':'Cement','SHREECEM.NS':'Cement','AMBUJACEM.NS':'Cement',
  'HDFCLIFE.NS':'Finance','SBILIFE.NS':'Finance','BAJAJFINSV.NS':'Finance',
  'BAJFINANCE.NS':'Finance','CHOLAFIN.NS':'Finance',
  'TITAN.NS':'Consumer','PIDILITIND.NS':'Consumer','ASIANPAINT.NS':'Consumer',
}

export function getSector(ticker: string): string { return SECTOR_MAP[ticker] || 'Other' }

export function colorVal(v: number | null | undefined, gt = 0): string {
  if (v == null) return 'var(--text-1)'
  return v > gt ? 'var(--green)' : v < gt ? 'var(--red)' : 'var(--text-1)'
}

export function fmt(v: number | null | undefined, dec = 2, suffix = ''): string {
  return v != null ? `${parseFloat(String(v)).toFixed(dec)}${suffix}` : '—'
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

export function getMarketStatus(): { label: string; class: string } {
  const now = new Date()
  const ist = new Date(now.toLocaleString('en-US', { timeZone: 'Asia/Kolkata' }))
  const h = ist.getHours()
  const m = ist.getMinutes()
  const day = ist.getDay()
  if (day === 0 || day === 6) return { label: 'Weekend', class: 'market-closed' }
  const mins = h * 60 + m
  if (mins < 555) return { label: 'Pre-Market', class: 'market-pre' }      // before 9:15
  if (mins < 930) return { label: 'Market Open', class: 'market-open' }     // 9:15 - 15:30
  return { label: 'After Hours', class: 'market-closed' }
}

export function getLogClass(msg: string): string {
  const l = msg.toLowerCase()
  if (l.includes('error') || l.includes('fail') || l.includes('crash')) return 'log-error'
  if (l.includes('warn') || l.includes('skip') || l.includes('404')) return 'log-warn'
  if (l.includes('✅') || l.includes('complete') || l.includes('done') || l.includes('success')) return 'log-success'
  return 'log-info'
}

// Chart.js shared config
export const gridOpts = { color: 'rgba(255,255,255,0.04)' }
export const tickOpts = { maxTicksLimit: 6 }
export const C = {
  blue: '#3b82f6', green: '#10b981', red: '#ef4444',
  purple: '#8b5cf6', amber: '#f59e0b', text: '#7c8aa0',
  bg: '#080c14', surf: '#121929', cyan: '#06b6d4',
}

// ── Hooks ──
const TAB_ORDER: TabId[] = ['research','signals','backtest','portfolio','journal','analytics','screener','settings']

export function useKeyboardShortcuts(setActiveTab: (t: TabId) => void) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return
      const n = parseInt(e.key)
      if (n >= 1 && n <= 8) { e.preventDefault(); setActiveTab(TAB_ORDER[n - 1]) }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [setActiveTab])
}

// ── Toast System ──
let toastId = 0
type ToastType = 'success' | 'error' | 'info'
type ToastItem = { id: number; type: ToastType; message: string }
let toastSetter: ((fn: (prev: ToastItem[]) => ToastItem[]) => void) | null = null

export function useToasts() {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  useEffect(() => {
    toastSetter = setToasts
  }, [])
  const remove = useCallback((id: number) => setToasts(prev => prev.filter(t => t.id !== id)), [])
  return { toasts, remove }
}

export function showToast(type: ToastType, message: string) {
  if (!toastSetter) return
  const id = ++toastId
  toastSetter(prev => [...prev, { id, type, message }])
  setTimeout(() => toastSetter?.(prev => prev.filter(t => t.id !== id)), 3000)
}

// Metric explanations
export const METRIC_TIPS: Record<string, string> = {
  'Total Return': 'Total cumulative return over the backtest period.',
  'Ann. Return': 'Annualized return, compounded over one year.',
  'Sharpe Ratio': 'Risk-adjusted return. >1 good, >2 excellent.',
  'Sortino Ratio': 'Like Sharpe but only penalizes downside volatility.',
  'Max Drawdown': 'Largest peak-to-trough decline during the period.',
  'Win Rate': 'Percentage of days with positive returns.',
  'Calmar Ratio': 'Annualized return divided by max drawdown.',
  'Omega Ratio': 'Probability-weighted ratio of gains vs losses. >1 is profitable.',
  'Information Ratio': 'Active return per unit of tracking error vs benchmark.',
  'CAPM Alpha (Ann.)': 'Excess return above what CAPM predicts, annualized.',
  'Beta': 'Sensitivity to market moves. 1.0 = moves with market.',
  'Tracking Error': 'Standard deviation of returns vs benchmark.',
  'Turnover Cost': 'Cumulative transaction costs from rebalancing (5bps per unit).',
  'Benchmark Return': 'Total return of NIFTY50 over same period.',
}
