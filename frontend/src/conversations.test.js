import test from 'node:test'
import assert from 'node:assert/strict'
import {
  loadStore, saveStore, createConversation, appendMessage, setSessionId, renameConversation,
  deleteConversation, togglePin, setActive, groupByDate, searchConversations, deriveTitle,
} from './conversations.js'

const fakeStorage = (init = {}) => {
  const m = new Map(Object.entries(init))
  return {
    getItem: k => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => void m.set(k, String(v)),
    removeItem: k => void m.delete(k),
    has: k => m.has(k),
    raw: k => m.get(k),
  }
}
const brokenStorage = () => ({ getItem() { throw new Error('denied') }, setItem() { throw new Error('full') }, removeItem() { throw new Error('denied') } })
const deepFreeze = o => { Object.values(o).forEach(v => v && typeof v === 'object' && deepFreeze(v)); return Object.freeze(o) }
const conv = (over = {}) => ({ ...createConversation(), ...over })
const user = content => ({ role: 'user', content })
const bot = content => ({ role: 'assistant', content })
const valid = s => s.items.length >= 1 && s.items.some(c => c.id === s.activeId)

test('createConversation returns a blank conversation; with a store it prepends and activates', () => {
  const a = createConversation(), b = createConversation()
  assert.notEqual(a.id, b.id)
  assert.deepEqual([a.title, a.pinned, a.titleLocked, a.sessionId, a.messages], ['New chat', false, false, null, []])
  const store = createConversation({ activeId: a.id, items: [a] })
  assert.equal(store.items.length, 2)
  assert.equal(store.activeId, store.items[0].id)
  assert.notEqual(store.activeId, a.id)
})

test('loadStore on empty storage yields a valid single-conversation store', () => {
  const s = loadStore('u1', fakeStorage())
  assert.equal(s.items.length, 1)
  assert.ok(valid(s))
})

test('migrates the v1 blob into one conversation and removes the legacy key', () => {
  const legacy = { messages: [user('hello there'), bot('hi')], sessionId: 'sess-1' }
  const st = fakeStorage({ 'research-agent-history-v1': JSON.stringify(legacy) })
  const s = loadStore('u1', st)
  assert.equal(s.items.length, 1)
  assert.equal(s.items[0].title, 'hello there')
  assert.equal(s.items[0].sessionId, 'sess-1')
  assert.deepEqual(s.items[0].messages, legacy.messages)
  assert.equal(st.has('research-agent-history-v1'), false)
  assert.equal(loadStore('u1', st).items[0].id, s.items[0].id) // persisted under the v2 key
})

test('migration: empty or corrupt legacy yields a blank chat and still clears the key; existing v2 wins', () => {
  for (const blob of [JSON.stringify({ messages: [], sessionId: 'x' }), '{nope']) {
    const st = fakeStorage({ 'research-agent-history-v1': blob })
    const s = loadStore(null, st)
    assert.equal(s.items[0].messages.length, 0)
    assert.equal(st.has('research-agent-history-v1'), false)
  }
  const st = fakeStorage({ 'research-agent-history-v1': JSON.stringify({ messages: [user('old')] }) })
  saveStore('u1', { activeId: 'k', items: [conv({ id: 'k', title: 'mine' })] }, st)
  assert.equal(loadStore('u1', st).items[0].title, 'mine')
  assert.ok(st.has('research-agent-history-v1')) // not touched when v2 already exists
})

test('corrupt or malformed stored JSON still produces a valid store', () => {
  for (const raw of ['{broken', 'null', '"str"', '{"items":"x"}', '{"items":[null,5]}', '{"activeId":"ghost","items":[{"id":"a"}]}']) {
    const s = loadStore('u1', fakeStorage({ 'research-agent-conversations-v2:u1': raw }))
    assert.ok(valid(s), raw)
  }
  assert.equal(loadStore('u1', fakeStorage({ 'research-agent-conversations-v2:u1': '{"activeId":"ghost","items":[{"id":"a"}]}' })).activeId, 'a')
})

test('unusable storage never throws', () => {
  assert.ok(valid(loadStore('u1', brokenStorage())))
  assert.ok(valid(loadStore('u1', null)))
  assert.equal(saveStore('u1', loadStore('u1', fakeStorage()), brokenStorage()), false)
})

test('history is isolated per user, with anon as the fallback', () => {
  const st = fakeStorage()
  const fresh = loadStore('alice', st)
  const a = appendMessage(fresh, fresh.activeId, user('alice secret'))
  saveStore('alice', a, st)
  assert.equal(JSON.stringify(loadStore('bob', st)).includes('alice secret'), false)
  assert.ok(JSON.stringify(loadStore('alice', st)).includes('alice secret'))
  saveStore(undefined, a, st)
  assert.ok(st.has('research-agent-conversations-v2:anon'))
})

test('saveStore caps conversations at 30, evicting oldest unpinned and keeping pinned', () => {
  const items = Array.from({ length: 35 }, (_, i) => conv({ id: `c${i}`, updatedAt: 1000 + i, pinned: i === 0 }))
  const st = fakeStorage()
  saveStore('u', { activeId: 'c34', items }, st)
  const s = loadStore('u', st)
  const ids = s.items.map(c => c.id)
  assert.equal(items.length, 35)
  assert.equal(ids.length, 30)
  assert.ok(ids.includes('c0'), 'oldest but pinned survives')
  for (const gone of ['c1', 'c2', 'c3', 'c4', 'c5']) assert.ok(!ids.includes(gone), gone)
  assert.ok(ids.includes('c6'))
})

test('eviction never drops the active conversation', () => {
  const items = Array.from({ length: 31 }, (_, i) => conv({ id: `c${i}`, updatedAt: 1000 + i }))
  const st = fakeStorage()
  saveStore('u', { activeId: 'c0', items }, st)
  const ids = loadStore('u', st).items.map(c => c.id)
  assert.ok(ids.includes('c0'))
  assert.ok(!ids.includes('c1'))
})

test('createConversation(store) applies the cap', () => {
  const items = Array.from({ length: 30 }, (_, i) => conv({ id: `c${i}`, updatedAt: 1000 + i }))
  const s = createConversation({ activeId: 'c29', items })
  assert.equal(s.items.length, 30)
  assert.ok(!s.items.some(c => c.id === 'c0'))
  assert.ok(valid(s))
})

test('messages are capped at the last 50 in appendMessage and saveStore', () => {
  let s = { activeId: 'a', items: [conv({ id: 'a' })] }
  for (let i = 0; i < 60; i++) s = appendMessage(s, 'a', user(`m${i}`))
  assert.equal(s.items[0].messages.length, 50)
  assert.equal(s.items[0].messages[0].content, 'm10')
  const st = fakeStorage()
  const big = { activeId: 'a', items: [conv({ id: 'a', messages: Array.from({ length: 80 }, (_, i) => user(`x${i}`)) })] }
  saveStore('u', big, st)
  const back = loadStore('u', st).items[0].messages
  assert.equal(back.length, 50)
  assert.equal(back[0].content, 'x30')
})

test('groupByDate buckets on local-day boundaries, pinned first, newest first, empties omitted', () => {
  const now = new Date(2026, 9, 15, 12, 0, 0).getTime()
  const at = (d, h = 0, m = 0, s = 0, ms = 0) => new Date(2026, 9, d, h, m, s, ms).getTime()
  const items = [
    conv({ id: 'today-start', updatedAt: at(15, 0, 0, 0, 0) }),
    conv({ id: 'today-late', updatedAt: at(15, 11, 59) }),
    conv({ id: 'yest-end', updatedAt: at(14, 23, 59, 59, 999) }),
    conv({ id: 'yest-start', updatedAt: at(14) }),
    conv({ id: 'wk-top', updatedAt: at(13, 23, 59) }),
    conv({ id: 'wk-edge', updatedAt: at(8) }),
    conv({ id: 'old-edge', updatedAt: at(8) - 1 }),
    conv({ id: 'pinned-old', updatedAt: at(1), pinned: true }),
    conv({ id: 'pinned-new', updatedAt: at(15, 9), pinned: true }),
  ]
  const g = groupByDate(items, now)
  assert.deepEqual(g.map(x => x.label), ['Pinned', 'Today', 'Yesterday', 'Previous 7 days', 'Older'])
  const ids = Object.fromEntries(g.map(x => [x.label, x.items.map(c => c.id)]))
  assert.deepEqual(ids.Pinned, ['pinned-new', 'pinned-old'])
  assert.deepEqual(ids.Today, ['today-late', 'today-start'])
  assert.deepEqual(ids.Yesterday, ['yest-end', 'yest-start'])
  assert.deepEqual(ids['Previous 7 days'], ['wk-top', 'wk-edge'])
  assert.deepEqual(ids.Older, ['old-edge'])
  assert.deepEqual(groupByDate([conv({ updatedAt: at(15, 1) })], now).map(x => x.label), ['Today'])
  assert.deepEqual(groupByDate([], now), [])
})

test('searchConversations matches title and message text case-insensitively', () => {
  const a = conv({ id: 'a', title: 'Transformers intro' })
  const b = conv({ id: 'b', title: 'Other', messages: [user('what about GANs?')] })
  const c = conv({ id: 'c', title: 'Nope' })
  const items = [a, b, c]
  assert.deepEqual(searchConversations(items, 'TRANSFORM').map(x => x.id), ['a'])
  assert.deepEqual(searchConversations(items, 'gans').map(x => x.id), ['b'])
  assert.equal(searchConversations(items, ''), items)
  assert.equal(searchConversations(items, '   '), items)
  assert.deepEqual(searchConversations(items, 'zzz'), [])
})

test('deriveTitle: first user message, whitespace collapsed, 48 chars with ellipsis, default when none', () => {
  assert.equal(deriveTitle([]), 'New chat')
  assert.equal(deriveTitle([bot('hi')]), 'New chat')
  assert.equal(deriveTitle([bot('hi'), user('  a \n\n  b\t c ')]), 'a b c')
  assert.equal(deriveTitle([user('   ')]), 'New chat')
  const exact = 'x'.repeat(48)
  assert.equal(deriveTitle([user(exact)]), exact)
  const long = deriveTitle([user('y'.repeat(100))])
  assert.equal([...long].length, 48)
  assert.ok(long.endsWith('…'))
})

test('title follows the first user message until renamed, then stays locked', () => {
  let s = { activeId: 'a', items: [conv({ id: 'a' })] }
  s = appendMessage(s, 'a', bot('welcome'))
  assert.equal(s.items[0].title, 'New chat')
  s = appendMessage(s, 'a', user('first question'))
  assert.equal(s.items[0].title, 'first question')
  s = appendMessage(s, 'a', user('second question'))
  assert.equal(s.items[0].title, 'first question')
  s = renameConversation(s, 'a', '  My notes  ')
  assert.deepEqual([s.items[0].title, s.items[0].titleLocked], ['My notes', true])
  s = appendMessage(s, 'a', user('third'))
  assert.equal(s.items[0].title, 'My notes')
  s = renameConversation(s, 'a', '   ')
  assert.deepEqual([s.items[0].title, s.items[0].titleLocked], ['first question', false])
  // renamed before any message: stays put
  let t = renameConversation({ activeId: 'b', items: [conv({ id: 'b' })] }, 'b', 'Custom')
  t = appendMessage(t, 'b', user('hello'))
  assert.equal(t.items[0].title, 'Custom')
})

test('appendMessage bumps updatedAt, setSessionId stores the thread, pin does not reorder', () => {
  let s = { activeId: 'a', items: [conv({ id: 'a', updatedAt: 5 })] }
  s = appendMessage(s, 'a', user('q'))
  assert.ok(s.items[0].updatedAt > 5)
  s = setSessionId(s, 'a', 'thread-9')
  assert.equal(s.items[0].sessionId, 'thread-9')
  const before = s.items[0].updatedAt
  s = togglePin(s, 'a')
  assert.deepEqual([s.items[0].pinned, s.items[0].updatedAt], [true, before])
  assert.equal(togglePin(s, 'a').items[0].pinned, false)
})

test('deleteConversation reselects the newest, and recreates a blank chat when emptied', () => {
  const s = { activeId: 'b', items: [conv({ id: 'a', updatedAt: 1 }), conv({ id: 'b', updatedAt: 2 }), conv({ id: 'c', updatedAt: 3 })] }
  assert.equal(deleteConversation(s, 'b').activeId, 'c')
  assert.equal(deleteConversation(s, 'a').activeId, 'b')
  const last = deleteConversation({ activeId: 'a', items: [conv({ id: 'a' })] }, 'a')
  assert.ok(valid(last))
  assert.notEqual(last.activeId, 'a')
})

test('setActive ignores unknown ids', () => {
  const s = { activeId: 'a', items: [conv({ id: 'a' }), conv({ id: 'b' })] }
  assert.equal(setActive(s, 'b').activeId, 'b')
  assert.equal(setActive(s, 'zzz'), s)
})

test('updaters are immutable', () => {
  const s = deepFreeze({ activeId: 'a', items: [conv({ id: 'a', messages: [user('hi')] }), conv({ id: 'b' })] })
  const snapshot = JSON.stringify(s)
  const outs = [
    appendMessage(s, 'a', bot('yo')), setSessionId(s, 'a', 'x'), renameConversation(s, 'a', 'T'),
    deleteConversation(s, 'a'), togglePin(s, 'a'), setActive(s, 'b'), createConversation(s),
  ]
  assert.equal(JSON.stringify(s), snapshot)
  outs.forEach(o => assert.notEqual(o, s))
  assert.equal(appendMessage(s, 'a', bot('yo')).items[1], s.items[1]) // untouched items are shared
  groupByDate(s.items, Date.now()); searchConversations(s.items, 'hi') // read-only on frozen input
})
