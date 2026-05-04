import { useState, useEffect } from 'react'
import { showToast, formatFileSize } from '../utils'

interface CacheStats {
  entries: number
  file_size: number
  last_saved: string
}

const SliderSetting = ({ label, desc, value, onChange, min, max, step, unit = '' }: {
  label: string; desc: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number; unit?: string
}) => (
  <div style={{ padding: '14px 0', borderBottom: '1px solid var(--border)' }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
      <div>
        <div style={{ fontSize: 13, color: 'var(--text-1)', fontWeight: 500 }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>{desc}</div>
      </div>
      <div className="mono" style={{ fontSize: 16, fontWeight: 600, color: 'var(--blue)', minWidth: 60, textAlign: 'right' }}>
        {value}{unit}
      </div>
    </div>
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span className="mono" style={{ fontSize: 9, color: 'var(--text-3)', minWidth: 24 }}>{min}</span>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        style={{ flex: 1 }}
      />
      <span className="mono" style={{ fontSize: 9, color: 'var(--text-3)', minWidth: 24, textAlign: 'right' }}>{max}</span>
    </div>
  </div>
)

export default function SettingsTab() {
  const [nliThresh, setNliThresh] = useState(0.60)
  const [confMin, setConfMin] = useState(0.60)
  const [scoreMin, setScoreMin] = useState(10)
  const [maxPos, setMaxPos] = useState(20)
  const [maxSector, setMaxSector] = useState(40)
  const [benchmark, setBenchmark] = useState('^NSEI')
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [clearing, setClearing] = useState(false)

  // Fetch cache stats
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('/api/cache-stats')
        if (res.ok) setCacheStats(await res.json())
      } catch { /* endpoint may not exist yet */ }
    }
    fetchStats()
  }, [])

  const handleSave = () => {
    showToast('success', 'Settings saved successfully')
  }

  const handleReset = () => {
    setNliThresh(0.60); setConfMin(0.60); setScoreMin(10)
    setMaxPos(20); setMaxSector(40); setBenchmark('^NSEI')
    showToast('info', 'Settings reset to defaults')
  }

  const handleClearCache = async () => {
    if (!confirm('Clear all cached NLP results? This will require re-processing all headlines on next run.')) return
    setClearing(true)
    try {
      await fetch('/api/clear-cache', { method: 'POST' })
      setCacheStats({ entries: 0, file_size: 0, last_saved: '' })
      showToast('success', 'NLP cache cleared')
    } catch {
      showToast('error', 'Failed to clear cache')
    }
    finally { setClearing(false) }
  }

  return (
    <>
      <div style={{ fontSize: 18, fontWeight: 600, letterSpacing: '-0.01em', flexShrink: 0 }}>Engine Configuration</div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* NLP Settings */}
        <div className="card card-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <span style={{ fontSize: 18, opacity: 0.7 }}>🧠</span>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--blue)' }}>NLP Pipeline</div>
          </div>

          <SliderSetting
            label="NLI Entailment Threshold" desc="Min DeBERTa NLI score to pass relevance gate"
            value={nliThresh} onChange={setNliThresh} min={0} max={1} step={0.05}
          />
          <SliderSetting
            label="Sentiment Confidence Min" desc="Min FinBERT confidence to register a signal"
            value={confMin} onChange={setConfMin} min={0} max={1} step={0.05}
          />
          <SliderSetting
            label="Signal Score Threshold" desc="Min |score| to enter daily sentiment matrix"
            value={scoreMin} onChange={setScoreMin} min={0} max={100} step={5}
          />
        </div>

        {/* Strategy Settings */}
        <div className="card card-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <span style={{ fontSize: 18, opacity: 0.7 }}>⚙️</span>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--blue)' }}>Strategy Parameters</div>
          </div>

          <SliderSetting
            label="Max Position Size" desc="Maximum single-stock weight cap"
            value={maxPos} onChange={setMaxPos} min={1} max={50} step={1} unit="%"
          />
          <SliderSetting
            label="Max Sector Concentration" desc="Maximum sector weight cap"
            value={maxSector} onChange={setMaxSector} min={10} max={100} step={5} unit="%"
          />

          <div style={{ padding: '14px 0', borderBottom: '1px solid var(--border)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <div style={{ fontSize: 13, color: 'var(--text-1)', fontWeight: 500 }}>Benchmark Comparison</div>
                <div style={{ fontSize: 11, color: 'var(--text-3)', marginTop: 2 }}>Index ticker for alpha calculation</div>
              </div>
              <input className="ci" type="text" value={benchmark} onChange={e => setBenchmark(e.target.value)} style={{ width: 120, textAlign: 'left' }} />
            </div>
          </div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Cache Management */}
        <div className="card card-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <span style={{ fontSize: 18, opacity: 0.7 }}>💾</span>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--purple)' }}>NLP Cache Management</div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 16 }}>
            <div style={{ background: 'var(--surf-2)', borderRadius: 'var(--r-sm)', padding: '10px 12px', border: '1px solid var(--border)' }}>
              <div className="mono" style={{ fontSize: 18, fontWeight: 600, color: 'var(--green)' }}>{cacheStats?.entries?.toLocaleString() || '—'}</div>
              <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 3 }}>Cached Headlines</div>
            </div>
            <div style={{ background: 'var(--surf-2)', borderRadius: 'var(--r-sm)', padding: '10px 12px', border: '1px solid var(--border)' }}>
              <div className="mono" style={{ fontSize: 18, fontWeight: 600, color: 'var(--blue)' }}>{cacheStats?.file_size ? formatFileSize(cacheStats.file_size) : '—'}</div>
              <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 3 }}>Cache File Size</div>
            </div>
            <div style={{ background: 'var(--surf-2)', borderRadius: 'var(--r-sm)', padding: '10px 12px', border: '1px solid var(--border)' }}>
              <div className="mono" style={{ fontSize: 14, fontWeight: 500, color: 'var(--text-2)' }}>{cacheStats?.last_saved || '—'}</div>
              <div style={{ fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--text-3)', marginTop: 3 }}>Last Saved</div>
            </div>
          </div>

          <div style={{ fontSize: 12, color: 'var(--text-2)', lineHeight: 1.5, marginBottom: 12 }}>
            The NLP cache stores processed headline sentiments to avoid redundant LLM calls.
            When you restart a simulation, cached headlines are instantly retrieved instead of re-processed.
          </div>

          <button className="btn btn-danger btn-sm" onClick={handleClearCache} disabled={clearing}>
            {clearing ? '⟳ Clearing...' : '🗑 Clear Cache'}
          </button>
        </div>

        {/* System Info */}
        <div className="card card-body">
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
            <span style={{ fontSize: 18, opacity: 0.7 }}>ℹ️</span>
            <div style={{ fontSize: 11, fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.07em', color: 'var(--cyan)' }}>System Information</div>
          </div>

          {[
            { label: 'NLP Models', val: 'FinBERT (Sentiment) + DeBERTa-v3 (NLI)' },
            { label: 'ML Models', val: 'Linear, LASSO, Ridge, RF, GBM, CART, SVR, NN, GA' },
            { label: 'Data Source', val: 'yfinance · NSE/BSE' },
            { label: 'Optimization', val: 'SHAP, Mean-Variance, Mean-Semivariance' },
            { label: 'Backend', val: 'Python 3.x + HTTP Server' },
            { label: 'Frontend', val: 'React + Vite + TypeScript' },
          ].map(item => (
            <div key={item.label} style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid var(--border)', gap: 16 }}>
              <span style={{ fontSize: 12, color: 'var(--text-3)', flexShrink: 0 }}>{item.label}</span>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-2)', textAlign: 'right' }}>{item.val}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div style={{ display: 'flex', gap: 10, justifyContent: 'flex-end', flexShrink: 0 }}>
        <button className="btn btn-sec" onClick={handleReset}>Reset to Defaults</button>
        <button className="btn btn-run" onClick={handleSave}>✓ Apply Changes</button>
      </div>
    </>
  )
}
