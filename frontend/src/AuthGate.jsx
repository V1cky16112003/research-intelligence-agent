import React, { useEffect, useState } from 'react'
import { supabase, authEnabled } from './auth.js'
import Login from './Login.jsx'

// Renders children only for a signed-in user (or always, when auth is not configured).
export default function AuthGate({ children }) {
  const [session, setSession] = useState(undefined)

  useEffect(() => {
    if (!authEnabled) return
    supabase.auth.getSession().then(({ data }) => setSession(data.session))
    const { data } = supabase.auth.onAuthStateChange((_event, s) => setSession(s))
    return () => data.subscription.unsubscribe()
  }, [])

  if (!authEnabled) return children({ user: null, signOut: null })
  if (session === undefined) return null
  if (!session) return <Login />
  return children({ user: session.user, signOut: () => supabase.auth.signOut() })
}
