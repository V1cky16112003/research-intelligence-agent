import { test, expect } from '@playwright/test'
import { sse, answerFor, jsonRoute, resetApp, openSidebar, ask, storeKey, conv } from './helpers.js'

let requests // bodies POSTed to /chat/stream, in order
let held // routes for query "slow", left hanging until a test releases them

test.beforeEach(async ({ page }) => {
  requests = []
  held = []
  await jsonRoute(page, '**/usage', { enabled: false })
  await page.route('**/chat/stream', route => {
    const body = route.request().postDataJSON()
    requests.push(body)
    if (body.query === 'slow') { held.push(route); return }
    route.fulfill({
      status: 200, contentType: 'text/event-stream',
      body: sse([['answer', answerFor(body.query)], ['done', {}]]),
    })
  })
  await resetApp(page)
})

// row actions are revealed on hover (always visible on touch), so hover first like a user would
const row = (nav, title) => nav.getByRole('listitem').filter({ hasText: title })
const action = async (nav, title, label) => {
  await row(nav, title).hover()
  await row(nav, title).getByLabel(label).click()
}

async function twoChats(page) {
  await ask(page, 'Alpha question')
  await expect(page.getByText('Answer to Alpha question')).toBeVisible()
  await openSidebar(page)
  await page.getByLabel('Start a new chat').click()
  await ask(page, 'Beta question')
  await expect(page.getByText('Answer to Beta question')).toBeVisible()
  return openSidebar(page)
}

test('chats are grouped under Today; switching restores messages and each chat keeps its session_id', async ({ page }) => {
  const nav = await twoChats(page)
  await expect(nav.getByRole('heading', { name: 'Today' })).toBeVisible()
  await expect(nav.getByRole('listitem')).toHaveCount(2)

  await nav.getByRole('button', { name: 'Alpha question' }).click()
  await expect(page.getByText('Answer to Alpha question')).toBeVisible()
  await expect(page.getByText('Answer to Beta question')).toHaveCount(0)

  await ask(page, 'Alpha follow-up')
  await expect(page.getByText('Answer to Alpha follow-up')).toBeVisible()
  expect(requests.map(r => [r.query, r.session_id])).toEqual([
    ['Alpha question', null],
    ['Beta question', null], // a new chat must not inherit Alpha's thread
    ['Alpha follow-up', 'sess-Alpha question'],
  ])
})

test('rename, pin, search and delete work from the sidebar and survive a reload', async ({ page }) => {
  let nav = await twoChats(page)

  // rename: double-click the title, type, Enter
  await nav.getByRole('button', { name: 'Alpha question' }).dblclick()
  await page.getByRole('textbox', { name: 'Rename chat' }).fill('Renamed chat')
  await page.keyboard.press('Enter')
  await expect(nav.getByRole('button', { name: 'Renamed chat' })).toBeVisible()

  // pin moves it under "Pinned"
  await action(nav, 'Renamed chat', 'Pin chat')
  const pinned = nav.locator('section').filter({ has: page.getByRole('heading', { name: 'Pinned' }) })
  await expect(pinned.getByRole('button', { name: 'Renamed chat' })).toBeVisible()

  // search matches title and message text
  const search = page.getByLabel('Search chats')
  await search.fill('beta')
  await expect(nav.getByRole('listitem')).toHaveCount(1)
  await search.fill('Answer to Alpha')
  await expect(nav.getByRole('button', { name: 'Renamed chat' })).toBeVisible()
  await expect(nav.getByRole('button', { name: 'Beta question' })).toHaveCount(0)
  await search.fill('zzz')
  await expect(nav.getByText('No chats match')).toBeVisible()
  await search.fill('')

  // reload keeps title, pin and messages
  await page.reload()
  nav = await openSidebar(page)
  await expect(nav.locator('section').filter({ has: page.getByRole('heading', { name: 'Pinned' }) })
    .getByRole('button', { name: 'Renamed chat' })).toBeVisible()
  await expect(nav.getByRole('button', { name: 'Beta question' })).toBeVisible()

  // delete asks inline first; No keeps it, Yes removes it
  await action(nav, 'Beta question', 'Delete chat')
  await expect(nav.getByText('Delete?')).toBeVisible()
  await nav.getByRole('button', { name: 'No' }).click()
  await expect(nav.getByRole('button', { name: 'Beta question' })).toBeVisible()
  await action(nav, 'Beta question', 'Delete chat')
  await nav.getByRole('button', { name: 'Yes' }).click()
  await expect(nav.getByRole('button', { name: 'Beta question' })).toHaveCount(0)
  await expect(nav.getByRole('listitem')).toHaveCount(1)
})

test('the legacy single-chat blob migrates into one conversation', async ({ page }) => {
  const legacy = {
    sessionId: 'legacy-s',
    messages: [
      { role: 'user', content: 'legacy question' },
      { role: 'assistant', content: 'legacy answer', citations: [], sqlResults: null, provider: 'groq' },
    ],
  }
  await resetApp(page, { 'research-agent-history-v1': legacy })
  await expect(page.getByText('legacy answer')).toBeVisible()
  const nav = await openSidebar(page)
  await expect(nav.getByRole('listitem')).toHaveCount(1)
  await expect(nav.getByRole('button', { name: 'legacy question' })).toBeVisible()
  expect(await page.evaluate(() => localStorage.getItem('research-agent-history-v1'))).toBeNull()

  await ask(page, 'next')
  await expect(page.getByText('Answer to next')).toBeVisible()
  expect(requests[0].session_id).toBe('legacy-s') // the migrated chat keeps its backend thread
})

test('switching chats mid-request aborts it and never leaks the answer into the other chat', async ({ page }) => {
  await ask(page, 'first')
  await expect(page.getByText('Answer to first')).toBeVisible()
  await openSidebar(page)
  await page.getByLabel('Start a new chat').click()
  await ask(page, 'slow')
  await expect.poll(() => held.length).toBe(1)

  const nav = await openSidebar(page)
  await nav.getByRole('button', { name: 'first', exact: true }).click()
  await expect(page.getByText('Answer to first')).toBeVisible()
  // the request is aborted; a late response must go nowhere
  await held[0].fulfill({
    status: 200, contentType: 'text/event-stream',
    body: sse([['answer', answerFor('slow', { answer: 'LEAKED' })], ['done', {}]]),
  }).catch(() => {})
  await expect(page.getByLabel('Send')).toBeVisible()
  await expect(page.getByText('LEAKED')).toHaveCount(0)

  await (await openSidebar(page)).getByRole('button', { name: 'slow' }).click()
  await expect(page.getByText('Stopped.')).toBeVisible()
  await expect(page.getByText('LEAKED')).toHaveCount(0)
})

test('history is namespaced per user id', async ({ page }) => {
  const other = { activeId: 'o1', items: [conv('o1', 'Someone elses chat', [{ role: 'user', content: 'private' }])] }
  await resetApp(page, { [storeKey('user-2')]: other })
  const nav = await openSidebar(page)
  await expect(nav.getByText('Someone elses chat')).toHaveCount(0)
  await expect(page.getByText('private')).toHaveCount(0)

  await resetApp(page, { [storeKey(null)]: { activeId: 'a1', items: [conv('a1', 'My chat', [{ role: 'user', content: 'mine' }])] }, [storeKey('user-2')]: other })
  await expect(page.getByText('mine', { exact: true })).toBeVisible()
  await expect(page.getByText('Someone elses chat')).toHaveCount(0)
})

test.describe('keyboard and accessibility', () => {
  test('every button has an accessible name', async ({ page }) => {
    await ask(page, 'Alpha question')
    await expect(page.getByText('Answer to Alpha question')).toBeVisible()
    await openSidebar(page)
    const unnamed = await page.evaluate(() => [...document.querySelectorAll('button')]
      .filter(b => !(b.getAttribute('aria-label') || b.textContent.trim() || b.title))
      .map(b => b.outerHTML.slice(0, 80)))
    expect(unnamed).toEqual([])
  })

  test('Esc closes the mobile drawer', async ({ page }, info) => {
    test.skip(info.project.name !== 'mobile', 'the drawer only exists under 768px')
    await openSidebar(page)
    await page.keyboard.press('Escape')
    await expect(page.getByRole('navigation', { name: 'Conversations' })).toBeHidden()
    await expect(page.getByRole('button', { name: 'Open sidebar' })).toBeVisible()
  })

  test('Cmd/Ctrl+K focuses search, also while the sidebar is collapsed', async ({ page }, info) => {
    const search = page.getByLabel('Search chats')
    if (info.project.name === 'desktop') {
      await page.keyboard.press('ControlOrMeta+k')
      await expect(search).toBeFocused()
      await page.getByRole('button', { name: 'Collapse sidebar' }).click()
      await expect(search).toBeHidden()
    }
    await page.keyboard.press('ControlOrMeta+k')
    await expect(search).toBeVisible()
    await expect(search).toBeFocused()
  })

  test('Cmd/Ctrl+Shift+O starts a new chat', async ({ page }) => {
    await ask(page, 'Alpha question')
    await expect(page.getByText('Answer to Alpha question')).toBeVisible()
    await page.getByLabel('Your question').blur()
    await page.keyboard.press('ControlOrMeta+Shift+O')
    await expect(page.getByRole('heading', { name: 'What would you like to research?' })).toBeVisible()
    await expect(page.getByText('Answer to Alpha question')).toHaveCount(0)
    const nav = await openSidebar(page)
    await expect(nav.getByRole('button', { name: 'Alpha question' })).toBeVisible()
  })

  test('Tab reaches the composer', async ({ page }) => {
    for (let i = 0; i < 40; i++) {
      await page.keyboard.press('Tab')
      if (await page.evaluate(() => document.activeElement?.getAttribute('aria-label') === 'Your question')) return
    }
    throw new Error('composer never received focus via Tab')
  })
})
