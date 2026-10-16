import { useCallback, useEffect, useRef, useState } from 'react'
import { authHeader } from './auth.js'

const API_URL = import.meta.env.VITE_API_URL || ''
const TICK_MS = 60_000

// Daily-question budget from GET /usage. Failures leave the last value in place and never throw.
export function useUsage() {
  const [usage, setUsage] = useState(null)
  const usageRef = useRef(usage)
  usageRef.current = usage
  const inflight = useRef(null)

  const refresh = useCallback(async () => {
    inflight.current?.abort()
    const ctl = (inflight.current = new AbortController())
    try {
      const res = await fetch(`${API_URL}/usage`, { signal: ctl.signal, headers: { ...(await authHeader()) } })
      if (!res.ok) return
      const data = await res.json()
      if (!ctl.signal.aborted) setUsage(data)
    } catch { /* network error or abort: keep what we have */ }
  }, [])

  // Optimistic: a 429 means the budget is gone, without waiting for the network.
  const markExhausted = useCallback(() => {
    setUsage(u => (u?.enabled ? { ...u, remaining: 0, used: u.limit, resets_in_seconds: u.resets_in_seconds || 3600 } : u))
  }, [])

  useEffect(() => {
    refresh()
    return () => inflight.current?.abort()
  }, [refresh])

  // Keep the reset countdown fresh locally; refetch once it reaches zero (the window has rolled over).
  useEffect(() => {
    const timer = setInterval(() => {
      const u = usageRef.current
      if (!u?.enabled || !(u.resets_in_seconds > 0)) return
      const next = Math.max(0, u.resets_in_seconds - TICK_MS / 1000)
      setUsage({ ...u, resets_in_seconds: next })
      if (next === 0) refresh()
    }, TICK_MS)
    return () => clearInterval(timer)
  }, [refresh])

  return { usage, refresh, markExhausted }
}
