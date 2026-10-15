import React, { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import './theme.css'
import './markdown.css'
import './app.css'
import { authHeader } from './auth.js'
import {
  loadStore, saveStore, createConversation, appendMessage, setSessionId, renameConversation,
  deleteConversation, togglePin, setActive, groupByDate, searchConversations,
} from './conversations.js'
import { useUsage } from './useUsage.js'
import Sidebar, { SidebarToggle } from './Sidebar.jsx'
import Message from './Message.jsx'
import RunTrace from './RunTrace.jsx'
import Composer from './Composer.jsx'
import Welcome from './Welcome.jsx'

const API_URL = import.meta.env.VITE_API_URL || ''
const COLLAPSED_KEY = 'research-agent-sidebar-collapsed'
const isMobile = () => window.matchMedia?.('(max-width: 767px)').matches

function initialCollapsed() {
  if (isMobile()) return true // the drawer starts closed
  try { return localStorage.getItem(COLLAPSED_KEY) === '1' } catch { return false }
}

// 429 / 503 carry a JSON {detail} from the API; anything else is a generic failure.
async function describeHttpError(res) {
  try {
    const body = await res.json()
    if (body?.detail && typeof body.detail === 'string') return body.detail
  } catch { /* not JSON */ }
  return `The API returned HTTP ${res.status}. It may be waking up — please try again in a moment.`
}

// Reads a text/event-stream body and calls onEvent(name, data) per event.
async function readSSE(res, onEvent) {
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const block = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      let name = 'message', data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: ')) name = line.slice(7)
        else if (line.startsWith('data: ')) data += line.slice(6)
      }
      if (data) onEvent(name, JSON.parse(data))
    }
  }
}

export default function App({ user, signOut }) {
  const userId = user?.id ?? null
  const [store, setStore] = useState(() => loadStore(userId))
  const [query, setQuery] = useState('')
  const [collapsed, setCollapsed] = useState(initialCollapsed)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [slowStart, setSlowStart] = useState(false)
  const [steps, setSteps] = useState([])
  const { usage, refresh } = useUsage()
  const abortRef = useRef(null)
  const scrollRef = useRef(null)
  const bottomRef = useRef(null)
  const nearBottom = useRef(true)

  const active = store.items.find(c => c.id === store.activeId) ?? store.items[0]
  const messages = active.messages

  useEffect(() => { saveStore(userId, store) }, [userId, store])

  useEffect(() => {
    try { localStorage.setItem(COLLAPSED_KEY, collapsed ? '1' : '0') } catch { /* best-effort */ }
  }, [collapsed])

  // Follow new output only while the reader is already at the bottom.
  useEffect(() => {
    if (nearBottom.current) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, steps])

  const onScroll = () => {
    const el = scrollRef.current
    if (el) nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  const groups = useMemo(() => {
    const items = searchConversations(store.items, query)
    // a blank untouched chat isn't history yet
    return groupByDate(items.filter(c => c.messages.length > 0))
  }, [store.items, query])

  const stopRun = useCallback(() => abortRef.current?.abort(), [])

  const newChat = useCallback(() => {
    stopRun()
    setInput('')
    // reuse the current chat if it is still empty instead of piling up blanks
    setStore(s => {
      const cur = s.items.find(c => c.id === s.activeId)
      return cur && cur.messages.length === 0 ? s : createConversation(s)
    })
  }, [stopRun])

  const selectChat = id => { stopRun(); setInput(''); setStore(s => setActive(s, id)) }

  const removeChat = id => {
    if (id === store.activeId) stopRun()
    setStore(s => deleteConversation(s, id))
  }

  useEffect(() => {
    function onKey(e) {
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.shiftKey && e.key.toLowerCase() === 'o') { e.preventDefault(); newChat() }
      // Sidebar focuses search itself, but its input is hidden while collapsed.
      if (mod && !e.shiftKey && e.key.toLowerCase() === 'k' && collapsed) {
        e.preventDefault()
        setCollapsed(false)
        setTimeout(() => document.querySelector('[aria-label="Search chats"]')?.focus(), 50)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [collapsed, newChat])

  async function sendMessage(text) {
    const q = text.trim()
    if (!q || loading) return
    const convId = active.id // a late event must land in the chat that asked
    const add = msg => setStore(s => appendMessage(s, convId, msg))

    add({ role: 'user', content: q })
    setInput('')
    setLoading(true)
    nearBottom.current = true

    // Show cold-start warning after 3s
    const slowTimer = setTimeout(() => setSlowStart(true), 3000)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        signal: controller.signal,
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', ...(await authHeader()) },
        body: JSON.stringify({ query: q, session_id: active.sessionId }),
      })
      clearTimeout(slowTimer)
      setSlowStart(false)

      if (!res.ok) {
        add({ role: 'assistant', content: await describeHttpError(res), citations: [], sqlResults: null, provider: 'error' })
        return
      }

      let answered = false
      await readSSE(res, (event, data) => {
        if (event === 'status') {
          setSteps(prev => [...prev, data.message])
        } else if (event === 'answer') {
          answered = true
          if (data.session_id) setStore(s => setSessionId(s, convId, data.session_id))
          add({
            role: 'assistant', content: data.answer, citations: data.citations,
            sqlResults: data.sql_results, provider: data.provider,
          })
        }
      })
      if (!answered) throw new Error('stream ended before an answer arrived')
    } catch (err) {
      clearTimeout(slowTimer)
      setSlowStart(false)
      add({
        role: 'assistant',
        content: err.name === 'AbortError'
          ? '*Stopped.*'
          : `Error: ${err.message}. The API may be waking up — please try again in a moment.`,
        citations: [], sqlResults: null, provider: 'error',
      })
    } finally {
      abortRef.current = null
      setLoading(false)
      setSteps([])
      refresh() // the server is the source of truth for what's left, incl. after a 429
    }
  }

  const greetingName = user?.user_metadata?.name?.split(' ')[0] || undefined

  return (
    <div className="app">
      <Sidebar
        groups={groups}
        activeId={store.activeId}
        query={query}
        onQuery={setQuery}
        onSelect={selectChat}
        onNew={newChat}
        onRename={(id, title) => setStore(s => renameConversation(s, id, title))}
        onDelete={removeChat}
        onPin={id => setStore(s => togglePin(s, id))}
        collapsed={collapsed}
        onToggle={() => setCollapsed(c => !c)}
        user={user}
        onSignOut={signOut ? () => { stopRun(); signOut() } : undefined}
        usage={usage}
      />

      <main className="app-main">
        <header className="app-bar">
          {collapsed && <SidebarToggle onClick={() => setCollapsed(false)} />}
          <h1 className="app-title">{active.title}</h1>
        </header>

        <div className="app-scroll" ref={scrollRef} onScroll={onScroll} role="log" aria-live="polite">
          <div className="app-col">
            {messages.length === 0 && <Welcome onPick={sendMessage} greetingName={greetingName} />}
            {messages.map((m, i) => <Message key={i} msg={m} />)}
            {slowStart && (
              <div className="app-cold">API is waking up from sleep — this may take 10-30s on first request...</div>
            )}
            {loading && <RunTrace steps={steps} active />}
            <div ref={bottomRef} />
          </div>
        </div>

        <div className="app-col app-composer">
          <Composer
            value={input}
            onChange={setInput}
            onSend={sendMessage}
            onStop={stopRun}
            loading={loading}
            usage={usage}
          />
        </div>
      </main>
    </div>
  )
}
