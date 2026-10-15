// Pure conversation-store logic (no React). Storage is touched only in loadStore/saveStore,
// always inside try/catch: it can be full, disabled, or hold corrupt JSON, and history is best-effort.

const PREFIX = 'research-agent-conversations-v2:'
const LEGACY_KEY = 'research-agent-history-v1'
const MAX_CONVERSATIONS = 30
const MAX_MESSAGES = 50
const TITLE_MAX = 48
const DEFAULT_TITLE = 'New chat'
const DAY_LABELS = ['Today', 'Yesterday', 'Previous 7 days', 'Older']

const storageKey = userId => PREFIX + (userId || 'anon')
const newId = () => globalThis.crypto?.randomUUID?.() ?? `c-${Date.now()}-${Math.random().toString(36).slice(2)}`

function defaultStorage() {
  try { return globalThis.localStorage ?? null } catch { return null }
}

export function deriveTitle(messages) {
  const first = (messages || []).find(m => m?.role === 'user')
  const text = String(first?.content ?? '').replace(/\s+/g, ' ').trim()
  if (!text) return DEFAULT_TITLE
  const chars = [...text]
  return chars.length > TITLE_MAX ? chars.slice(0, TITLE_MAX - 1).join('') + '…' : text
}

// No arg: a bare conversation. With a store: that store plus a new active conversation (cap applied).
export function createConversation(store) {
  const now = Date.now()
  const c = {
    id: newId(), title: DEFAULT_TITLE, titleLocked: false, pinned: false,
    createdAt: now, updatedAt: now, sessionId: null, messages: [],
  }
  return store ? cap({ activeId: c.id, items: [c, ...store.items] }) : c
}

// Trim messages and evict the oldest unpinned conversations beyond the cap (never the active one).
function cap(store) {
  let items = store.items.map(c => c.messages.length > MAX_MESSAGES ? { ...c, messages: c.messages.slice(-MAX_MESSAGES) } : c)
  const extra = items.length - MAX_CONVERSATIONS
  if (extra > 0) {
    const drop = new Set(
      items.filter(c => c.id !== store.activeId)
        .sort((a, b) => a.pinned - b.pinned || a.updatedAt - b.updatedAt)
        .slice(0, extra).map(c => c.id),
    )
    items = items.filter(c => !drop.has(c.id))
  }
  return { ...store, items }
}

function normItem(c) {
  const messages = Array.isArray(c.messages) ? c.messages.slice(-MAX_MESSAGES) : []
  const now = Date.now()
  return {
    id: typeof c.id === 'string' && c.id ? c.id : newId(),
    title: typeof c.title === 'string' ? c.title : deriveTitle(messages),
    titleLocked: !!c.titleLocked,
    pinned: !!c.pinned,
    createdAt: Number.isFinite(c.createdAt) ? c.createdAt : now,
    updatedAt: Number.isFinite(c.updatedAt) ? c.updatedAt : now,
    sessionId: c.sessionId ?? null,
    messages,
  }
}

// Always yields a valid store: at least one conversation, activeId pointing at one of them.
function normalize(raw) {
  const items = (Array.isArray(raw?.items) ? raw.items : []).filter(c => c && typeof c === 'object').map(normItem)
  if (!items.length) items.push(createConversation())
  return { activeId: items.some(c => c.id === raw?.activeId) ? raw.activeId : items[0].id, items }
}

export function saveStore(userId, store, storage = defaultStorage()) {
  try {
    storage.setItem(storageKey(userId), JSON.stringify(cap(store)))
    return true
  } catch {
    return false
  }
}

function migrateLegacy(userId, storage) {
  try {
    const raw = storage.getItem(LEGACY_KEY)
    if (raw === null) return normalize(null)
    let legacy = null
    try { legacy = JSON.parse(raw) } catch { /* corrupt legacy blob: drop it */ }
    const messages = Array.isArray(legacy?.messages) ? legacy.messages : []
    const store = normalize(messages.length ? { items: [{ messages, sessionId: legacy.sessionId }] } : null)
    if (saveStore(userId, store, storage)) storage.removeItem(LEGACY_KEY)
    return store
  } catch {
    return normalize(null)
  }
}

export function loadStore(userId, storage = defaultStorage()) {
  try {
    const raw = storage.getItem(storageKey(userId))
    if (raw !== null) return normalize(JSON.parse(raw))
  } catch {
    return normalize(null)
  }
  return migrateLegacy(userId, storage)
}

const patch = (store, id, fn) => ({ ...store, items: store.items.map(c => (c.id === id ? fn(c) : c)) })

// The title follows the first user message until the user renames (titleLocked). Only the
// default title is re-derived, so it can't drift once trimming drops the original first message.
export function appendMessage(store, id, msg) {
  return patch(store, id, c => {
    const messages = [...c.messages, msg].slice(-MAX_MESSAGES)
    const title = c.titleLocked || c.title !== DEFAULT_TITLE ? c.title : deriveTitle(messages)
    return { ...c, messages, title, updatedAt: Date.now() }
  })
}

export const setSessionId = (store, id, sessionId) => patch(store, id, c => ({ ...c, sessionId }))

// A blank rename unlocks the title and re-derives it.
export function renameConversation(store, id, title) {
  const t = String(title ?? '').trim().slice(0, 100)
  return patch(store, id, c => (t
    ? { ...c, title: t, titleLocked: true }
    : { ...c, title: deriveTitle(c.messages), titleLocked: false }))
}

export function deleteConversation(store, id) {
  let items = store.items.filter(c => c.id !== id)
  if (!items.length) items = [createConversation()]
  const activeId = items.some(c => c.id === store.activeId)
    ? store.activeId
    : [...items].sort((a, b) => b.updatedAt - a.updatedAt)[0].id
  return { activeId, items }
}

export const togglePin = (store, id) => patch(store, id, c => ({ ...c, pinned: !c.pinned }))

export const setActive = (store, id) => (store.items.some(c => c.id === id) ? { ...store, activeId: id } : store)

export function groupByDate(items, now = Date.now()) {
  const start = new Date(now)
  start.setHours(0, 0, 0, 0)
  const day = n => { const d = new Date(start); d.setDate(d.getDate() - n); return d.getTime() }
  const [today, yesterday, week] = [day(0), day(1), day(7)]
  const groups = { Pinned: [], Today: [], Yesterday: [], 'Previous 7 days': [], Older: [] }
  for (const c of items) {
    const label = c.pinned ? 'Pinned'
      : c.updatedAt >= today ? 'Today'
        : c.updatedAt >= yesterday ? 'Yesterday'
          : c.updatedAt >= week ? 'Previous 7 days' : 'Older'
    groups[label].push(c)
  }
  return ['Pinned', ...DAY_LABELS]
    .map(label => ({ label, items: groups[label].sort((a, b) => b.updatedAt - a.updatedAt) }))
    .filter(g => g.items.length)
}

export function searchConversations(items, q) {
  const needle = String(q ?? '').trim().toLowerCase()
  if (!needle) return items
  return items.filter(c =>
    String(c.title).toLowerCase().includes(needle)
    || c.messages.some(m => String(m?.content ?? '').toLowerCase().includes(needle)))
}
