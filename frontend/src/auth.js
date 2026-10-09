import { createClient } from '@supabase/supabase-js'

// Auth is on when the build has Supabase config; otherwise the app runs open
// (local dev against a backend without SUPABASE_URL).
const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

export const supabase = url && anonKey ? createClient(url, anonKey) : null
export const authEnabled = Boolean(supabase)

// Comma-separated OAuth providers enabled in the Supabase dashboard, e.g. "github,google".
export const oauthProviders = (import.meta.env.VITE_SUPABASE_OAUTH_PROVIDERS ?? 'github')
  .split(',').map(p => p.trim()).filter(Boolean)

export async function authHeader() {
  if (!supabase) return {}
  // getSession refreshes an expired access token before returning it.
  const { data } = await supabase.auth.getSession()
  const token = data.session?.access_token
  return token ? { Authorization: `Bearer ${token}` } : {}
}
