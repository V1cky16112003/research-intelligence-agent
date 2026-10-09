import { test, expect } from '@playwright/test'

// supabase-js persists the session under sb-<project-ref>-auth-token; for
// http://localhost:54321 the ref is "localhost".
const STORAGE_KEY = 'sb-localhost-auth-token'

function fakeSession() {
  const b64 = o => Buffer.from(JSON.stringify(o)).toString('base64url')
  const exp = Math.floor(Date.now() / 1000) + 3600
  const access_token = `${b64({ alg: 'ES256', typ: 'JWT' })}.${b64({ sub: 'user-1', exp, aud: 'authenticated', email: 'a@b.c' })}.sig`
  return { access_token, refresh_token: 'r', token_type: 'bearer', expires_in: 3600, expires_at: exp, user: { id: 'user-1', email: 'a@b.c', aud: 'authenticated' } }
}

test('signed-out visitors see the sign-in screen', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('button', { name: 'Continue with Github' })).toBeVisible()
  await expect(page.getByLabel('Email address')).toBeVisible()
  await expect(page.getByLabel('Your question')).toHaveCount(0)
})

test('signed-in users send their access token and can sign out', async ({ page }) => {
  const session = fakeSession()
  await page.addInitScript(([k, v]) => {
    if (!sessionStorage.getItem('seeded')) { localStorage.setItem(k, v); sessionStorage.setItem('seeded', '1') }
  }, [STORAGE_KEY, JSON.stringify(session)])
  await page.route('http://localhost:54321/**', route => route.fulfill({ status: 204, body: '' }))

  let auth
  await page.route('**/chat/stream', route => {
    auth = route.request().headers()['authorization']
    route.fulfill({
      status: 200, contentType: 'text/event-stream',
      body: `event: answer\ndata: ${JSON.stringify({ answer: 'hello', citations: [], sql_results: null, session_id: 'user-1:x', provider: 'groq' })}\n\nevent: done\ndata: {}\n\n`,
    })
  })

  await page.goto('/')
  await page.getByLabel('Your question').fill('q')
  await page.getByLabel('Send').click()
  await expect(page.getByText('hello')).toBeVisible()
  expect(auth).toBe(`Bearer ${session.access_token}`)

  await page.getByLabel('Sign out').click()
  await expect(page.getByRole('button', { name: 'Continue with Github' })).toBeVisible()
})
