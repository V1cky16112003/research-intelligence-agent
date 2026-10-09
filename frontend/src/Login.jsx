import React, { useState } from 'react'
import { supabase, oauthProviders } from './auth.js'

const box = { maxWidth: 380, margin: '12vh auto', padding: 28, background: '#141414', border: '1px solid #2a2a2a', borderRadius: 14, color: '#ddd' }
const btn = { width: '100%', padding: '11px 14px', borderRadius: 10, border: '1px solid #333', background: '#1e3a5f', color: '#fff', fontSize: 15, cursor: 'pointer', marginTop: 10 }
const input = { width: '100%', padding: '11px 14px', borderRadius: 10, border: '1px solid #2a2a2a', background: '#0f0f0f', color: '#e8e8e8', fontSize: 15, marginTop: 10 }
const label = name => name.charAt(0).toUpperCase() + name.slice(1)

export default function Login() {
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState(null)
  const [busy, setBusy] = useState(false)

  async function oauth(provider) {
    setBusy(true)
    const { error } = await supabase.auth.signInWithOAuth({ provider, options: { redirectTo: window.location.origin } })
    if (error) { setStatus({ error: error.message }); setBusy(false) }
  }

  async function magicLink(e) {
    e.preventDefault()
    if (!email.trim()) return
    setBusy(true)
    const { error } = await supabase.auth.signInWithOtp({ email: email.trim(), options: { emailRedirectTo: window.location.origin } })
    setBusy(false)
    setStatus(error ? { error: error.message } : { ok: `Check ${email.trim()} for a sign-in link.` })
  }

  return (
    <main style={box}>
      <h1 style={{ fontSize: 20, color: '#fff' }}>Research Intelligence Agent</h1>
      <p style={{ fontSize: 14, color: '#888', margin: '6px 0 14px' }}>Sign in to ask about 50k ArXiv ML papers.</p>
      {oauthProviders.map(p => (
        <button key={p} style={btn} disabled={busy} onClick={() => oauth(p)}>Continue with {label(p)}</button>
      ))}
      <form onSubmit={magicLink} style={{ marginTop: 18, borderTop: '1px solid #222', paddingTop: 8 }}>
        <input style={input} type="email" aria-label="Email address" placeholder="you@example.com"
          value={email} onChange={e => setEmail(e.target.value)} required />
        <button style={{ ...btn, background: 'transparent' }} disabled={busy} type="submit">Email me a sign-in link</button>
      </form>
      {status && (
        <p role="status" style={{ marginTop: 14, fontSize: 13, color: status.error ? '#e07070' : '#7fbf7f' }}>
          {status.error || status.ok}
        </p>
      )}
    </main>
  )
}
