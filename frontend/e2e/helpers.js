// Shared by the history/usage specs. Not a spec itself (playwright only runs *.spec.js).

export const sse = events => events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join('')

export const answerFor = (query, extra = {}) => ({
  answer: `Answer to ${query}`, citations: [], sql_results: null, session_id: `sess-${query}`, provider: 'groq', ...extra,
})

export const jsonRoute = (page, glob, body, status = 200) =>
  page.route(glob, route => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) }))

// Fresh storage, then reload so the app boots from it.
export async function resetApp(page, seed = {}) {
  await page.goto('/')
  await page.evaluate(seed => {
    localStorage.clear()
    for (const [k, v] of Object.entries(seed)) localStorage.setItem(k, typeof v === 'string' ? v : JSON.stringify(v))
  }, seed)
  await page.reload()
}

// On mobile the sidebar is a drawer that starts closed; on desktop it is already open.
export async function openSidebar(page) {
  const open = page.getByRole('button', { name: 'Open sidebar' })
  if (await open.isVisible()) await open.click()
  const nav = page.getByRole('navigation', { name: 'Conversations' })
  await nav.waitFor({ state: 'visible' })
  return nav
}

export async function ask(page, text) {
  const close = page.getByRole('button', { name: 'Close sidebar' }) // an open mobile drawer covers the composer
  // 'Open sidebar' is rendered exactly when collapsed, so it is a race-free signal that the drawer is not mid-close
  if (!(await page.getByRole('button', { name: 'Open sidebar' }).isVisible()) && (await close.isVisible())) await close.click()
  await page.getByLabel('Your question').fill(text)
  await page.getByLabel('Send').click()
}

export const storeKey = id => `research-agent-conversations-v2:${id ?? 'anon'}`

export const conv = (id, title, messages = [], extra = {}) => ({
  id, title, titleLocked: false, pinned: false, createdAt: Date.now(), updatedAt: Date.now(), sessionId: null, messages, ...extra,
})
