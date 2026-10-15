import { test, expect } from '@playwright/test'
import { sse, answerFor, jsonRoute, resetApp, openSidebar, ask } from './helpers.js'

const USAGE = { enabled: true, limit: 100, used: 27, remaining: 73, resets_in_seconds: 21540, per_minute_limit: 20 }

// Serves usage.body (tests swap it mid-run) and counts the calls.
async function mockUsage(page, body) {
  const usage = { n: 0, body }
  await page.route('**/usage', route => {
    usage.n++
    route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(usage.body) })
  })
  return usage
}

const answerStream = page => page.route('**/chat/stream', route => route.fulfill({
  status: 200, contentType: 'text/event-stream',
  body: sse([['answer', answerFor(route.request().postDataJSON().query)], ['done', {}]]),
}))

test('shows the remaining questions and reset time with a progressbar', async ({ page }) => {
  await mockUsage(page, USAGE)
  await resetApp(page)
  await openSidebar(page)
  await expect(page.getByText('73 / 100 left')).toBeVisible()
  await expect(page.getByText('resets in 5h 59m')).toBeVisible()
  const bar = page.getByRole('progressbar', { name: 'Questions left today' })
  await expect(bar).toHaveAttribute('aria-valuenow', '73')
  await expect(bar).toHaveAttribute('aria-valuemax', '100')
})

test('renders no meter when usage is disabled', async ({ page }) => {
  await mockUsage(page, { enabled: false })
  await resetApp(page)
  await openSidebar(page)
  await expect(page.getByRole('progressbar')).toHaveCount(0)
  await expect(page.getByText(/ left/)).toHaveCount(0)
})

test('out of questions disables Send even with text typed', async ({ page }) => {
  await mockUsage(page, { ...USAGE, used: 100, remaining: 0, resets_in_seconds: 3 * 3600 })
  await resetApp(page)
  await openSidebar(page)
  await expect(page.getByText(/^Out of questions/)).toBeVisible()
  await page.getByLabel('Your question').fill('anything')
  await expect(page.getByLabel('Send')).toBeDisabled()
  await expect(page.getByRole('status').filter({ hasText: 'out of questions for today' })).toBeVisible()
})

test('low budget switches the meter to the danger colour', async ({ page }) => {
  await mockUsage(page, { ...USAGE, used: 85, remaining: 15 })
  await resetApp(page)
  await openSidebar(page)
  await expect(page.getByText('15 / 100 left')).toBeVisible()
  const fill = page.locator('.um-fill')
  const danger = await page.evaluate(() => {
    const probe = document.createElement('i')
    probe.style.color = 'var(--danger)'
    document.body.append(probe)
    const c = getComputedStyle(probe).color
    probe.remove()
    return c
  })
  await expect(fill).toHaveCSS('background-color', danger)

  // and a healthy budget does not use it
  await jsonRoute(page, '**/usage', USAGE)
  await page.reload()
  await openSidebar(page)
  await expect(page.getByText('73 / 100 left')).toBeVisible()
  await expect(fill).not.toHaveCSS('background-color', danger)
})

test('usage is refetched after a completed chat request', async ({ page }) => {
  const calls = await mockUsage(page, USAGE)
  await answerStream(page)
  await resetApp(page)
  await openSidebar(page)
  await expect(page.getByText('73 / 100 left')).toBeVisible()
  const before = calls.n
  calls.body = { ...USAGE, used: 28, remaining: 72 }

  await ask(page, 'hello')
  await expect(page.getByText('Answer to hello')).toBeVisible()
  await expect.poll(() => calls.n).toBeGreaterThan(before)
  await openSidebar(page)
  await expect(page.getByText('72 / 100 left')).toBeVisible()
})

test('a 429 shows the API detail and refetches usage', async ({ page }) => {
  const calls = await mockUsage(page, USAGE)
  await page.route('**/chat/stream', route => route.fulfill({
    status: 429, contentType: 'application/json', body: JSON.stringify({ detail: 'Daily limit of 100 questions reached.' }),
  }))
  await resetApp(page)
  await openSidebar(page)
  await expect(page.getByText('73 / 100 left')).toBeVisible() // initial fetch landed before the body is swapped
  const before = calls.n
  calls.body = { ...USAGE, used: 100, remaining: 0, resets_in_seconds: 7200 }

  await ask(page, 'one too many')
  await expect(page.getByText('Daily limit of 100 questions reached.')).toBeVisible()
  await expect.poll(() => calls.n).toBeGreaterThan(before)
  await expect(page.getByLabel('Your question')).toHaveValue('') // the question was sent, not stuck in the box
  await openSidebar(page)
  await expect(page.getByText(/^Out of questions/)).toBeVisible()
})
