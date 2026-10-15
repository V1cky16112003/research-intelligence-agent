import React, { useLayoutEffect, useRef } from 'react'
import { formatReset } from './UsageMeter.jsx'
import './chat.css'

const MAX_LEN = 4000
const MAX_HEIGHT = 8 * 24 + 24 // 8 rows of 24px line-height + vertical padding

export default function Composer({ value, onChange, onSend, onStop, loading, usage, disabledReason }) {
  const ref = useRef(null)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`
  }, [value])

  const outOfQuota = usage?.enabled && usage.remaining === 0
  const notice = disabledReason
    || (outOfQuota ? `You are out of questions for today — resets in ${formatReset(usage.resets_in_seconds ?? 0)}` : null)
  const canSend = !!value.trim() && !notice

  function submit() {
    if (canSend && !loading) onSend(value)
  }

  return (
    <div className="composer">
      {notice && <div className="composer-notice" role="status">{notice}</div>}
      <div className="composer-box">
        <textarea
          ref={ref}
          className="composer-input"
          value={value}
          onChange={e => onChange(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault()
              submit()
            }
          }}
          placeholder="Ask about ML papers…"
          rows={1}
          aria-label="Your question"
          maxLength={MAX_LEN}
        />
        {loading ? (
          <button className="composer-btn stop" onClick={onStop} aria-label="Stop generating">■</button>
        ) : (
          <button className="composer-btn" onClick={submit} disabled={!canSend} aria-label="Send">↑</button>
        )}
      </div>
      {value.length > 3500 && <div className="composer-count">{value.length} / {MAX_LEN}</div>}
    </div>
  )
}
