import React from 'react'

// shortcut: seconds → "<1m" | "12m" | "6h 12m"; no days, the quota window is 24h max.
export function formatReset(seconds) {
  const m = Math.floor(Math.max(0, seconds || 0) / 60)
  if (m < 1) return '<1m'
  if (m < 60) return `${m}m`
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, '0')}m`
}

export default function UsageMeter({ usage }) {
  if (!usage || !usage.enabled) return null
  const limit = Math.max(0, usage.limit || 0)
  const remaining = Math.min(limit, Math.max(0, usage.remaining ?? 0))
  const pct = limit > 0 ? (remaining / limit) * 100 : 0
  const low = pct < 20
  const reset = `resets in ${formatReset(usage.resets_in_seconds)}`

  return (
    <div className="um" data-low={low || undefined}>
      <div
        className="um-track"
        role="progressbar"
        aria-label="Questions left today"
        aria-valuemin={0}
        aria-valuemax={limit}
        aria-valuenow={remaining}
        aria-valuetext={`${remaining} of ${limit} questions left`}
      >
        <div className="um-fill" style={{ width: `${pct}%` }} />
      </div>
      <div className="um-text">
        {remaining === 0 ? (
          <span className="um-out">Out of questions · {reset}</span>
        ) : (
          <>
            <span className="um-num">{remaining} / {limit} left</span>
            <span className="um-reset">{reset}</span>
          </>
        )}
      </div>
    </div>
  )
}
