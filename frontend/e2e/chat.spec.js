import { test, expect } from '@playwright/test'

const sse = events => events.map(([name, data]) => `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`).join('')

const ANSWER = {
  answer: '## Result\n\nThe corpus holds **50,000 papers**. See arXiv:1706.03762.',
  citations: [],
  sql_results: [{ total_papers: 50000 }],
  session_id: 'sess-1',
  provider: 'groq',
}

async function mockStream(page, body, status = 200, contentType = 'text/event-stream') {
  await page.route('**/chat/stream', route => route.fulfill({ status, contentType, body }))
}

test.beforeEach(async ({ page }) => {
  await page.goto('/')
  await page.evaluate(() => localStorage.clear())
  await page.reload()
})

test('streams progress then renders the markdown answer and SQL table', async ({ page }) => {
  await mockStream(page, sse([
    ['status', { node: 'planner', message: 'Planned: sql_analytics' }],
    ['status', { node: 'reporter', message: 'Drafted the answer' }],
    ['answer', ANSWER],
    ['done', {}],
  ]))

  await page.getByLabel('Your question').fill('How many papers?')
  await page.getByLabel('Send').click()

  await expect(page.getByRole('heading', { name: 'Result' })).toBeVisible()
  await expect(page.getByText('50,000 papers')).toBeVisible()
  await expect(page.getByRole('cell', { name: '50,000' })).toBeVisible()
  await expect(page.getByText('via groq')).toBeVisible()
})

test('example query button sends the question', async ({ page }) => {
  let sent
  await page.route('**/chat/stream', route => {
    sent = route.request().postDataJSON()
    route.fulfill({ status: 200, contentType: 'text/event-stream', body: sse([['answer', ANSWER], ['done', {}]]) })
  })
  await page.getByRole('button', { name: /attention mechanisms/ }).click()
  await expect(page.getByText('50,000 papers')).toBeVisible()
  expect(sent.query).toContain('attention mechanisms')
})

test('shows the API detail on 503 instead of a generic error', async ({ page }) => {
  await mockStream(page, JSON.stringify({ detail: 'The agent is busy with other questions — please retry in a few seconds.' }), 503, 'application/json')
  await page.getByLabel('Your question').fill('q')
  await page.getByLabel('Send').click()
  await expect(page.getByText(/agent is busy/)).toBeVisible()
})

test('stop button aborts an in-flight request', async ({ page }) => {
  await page.route('**/chat/stream', () => { /* never fulfilled: request hangs */ })
  await page.getByLabel('Your question').fill('slow question')
  await page.getByLabel('Send').click()
  await page.getByLabel('Stop generating').click()
  await expect(page.getByText('Stopped.')).toBeVisible()
  await expect(page.getByLabel('Send')).toBeVisible()
})

test('history survives a reload and New chat clears it', async ({ page }) => {
  await mockStream(page, sse([['answer', ANSWER], ['done', {}]]))
  await page.getByLabel('Your question').fill('How many papers?')
  await page.getByLabel('Send').click()
  await expect(page.getByText('50,000 papers')).toBeVisible()

  await page.reload()
  await expect(page.getByText('50,000 papers')).toBeVisible()

  await page.getByLabel('Start a new chat').click()
  await expect(page.getByText('Try an example:')).toBeVisible()
})

test('copy button copies the answer', async ({ page, context, browserName }) => {
  test.skip(browserName !== 'chromium')
  await context.grantPermissions(['clipboard-read', 'clipboard-write'])
  await mockStream(page, sse([['answer', ANSWER], ['done', {}]]))
  await page.getByLabel('Your question').fill('q')
  await page.getByLabel('Send').click()
  await page.getByLabel('Copy answer').click()
  await expect(page.getByLabel('Copy answer')).toHaveText('Copied')
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('50,000 papers')
})
