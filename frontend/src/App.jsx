import React, { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './markdown.css'
import { authHeader } from './auth.js'

const API_URL = import.meta.env.VITE_API_URL || ''
const HISTORY_KEY = 'research-agent-history-v1'
const HISTORY_LIMIT = 50

function loadHistory() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || 'null')
    return saved && Array.isArray(saved.messages) ? saved : { messages: [], sessionId: null }
  } catch {
    return { messages: [], sessionId: null }
  }
}

// 429 / 503 carry a JSON {detail} from the API; anything else is a generic failure.
async function describeHttpError(res) {
  try {
    const body = await res.json()
    if (body?.detail && typeof body.detail === 'string') return body.detail
  } catch { /* not JSON */ }
  return `The API returned HTTP ${res.status}. It may be waking up — please try again in a moment.`
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      style={styles.copyBtn}
      aria-label="Copy answer"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true)
          setTimeout(() => setCopied(false), 1500)
        })
      }}
    >
      {copied ? 'Copied' : 'Copy'}
    </button>
  )
}

// --- Styles (inline for simplicity, no CSS file needed) ---
const styles = {
  container: { display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '900px', margin: '0 auto', padding: '0 16px' },
  header: { padding: '20px 0 12px', borderBottom: '1px solid #2a2a2a' },
  title: { fontSize: '20px', fontWeight: 600, color: '#fff' },
  subtitle: { fontSize: '13px', color: '#666', marginTop: '4px' },
  providerBadge: { display: 'inline-block', fontSize: '11px', padding: '2px 8px', borderRadius: '12px', marginLeft: '8px', background: '#1a1a2e', color: '#6c8ebf' },
  messages: { flex: 1, overflowY: 'auto', padding: '20px 0', display: 'flex', flexDirection: 'column', gap: '16px' },
  userMsg: { alignSelf: 'flex-end', background: '#1e3a5f', padding: '12px 16px', borderRadius: '16px 16px 4px 16px', maxWidth: '75%', fontSize: '15px', lineHeight: '1.5', whiteSpace: 'pre-wrap' },
  assistantMsg: { alignSelf: 'flex-start', background: '#1a1a1a', padding: '16px 20px', borderRadius: '16px 16px 16px 4px', maxWidth: '92%', width: 'fit-content', fontSize: '15px', lineHeight: '1.6', border: '1px solid #2a2a2a' },
  citations: { marginTop: '12px', padding: '10px', background: '#111', borderRadius: '8px', fontSize: '13px', border: '1px solid #222' },
  citationTitle: { color: '#888', marginBottom: '6px', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' },
  citationItem: { padding: '6px 0', borderBottom: '1px solid #1a1a1a', color: '#aaa' },
  sqlResults: { marginTop: '12px', background: '#0d1219', borderRadius: '8px', border: '1px solid #1a2a4a', maxHeight: '320px', overflowY: 'auto' },
  loadingDots: { alignSelf: 'flex-start', padding: '12px 16px', background: '#1a1a1a', borderRadius: '16px', border: '1px solid #2a2a2a' },
  coldStartBanner: { background: '#1a1a00', border: '1px solid #333300', borderRadius: '8px', padding: '10px 14px', fontSize: '13px', color: '#aaaa00', marginBottom: '12px', textAlign: 'center' },
  inputArea: { padding: '16px 0 24px', borderTop: '1px solid #2a2a2a', display: 'flex', gap: '8px' },
  input: { flex: 1, background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '12px', padding: '12px 16px', color: '#e8e8e8', fontSize: '15px', outline: 'none', resize: 'none' },
  sendBtn: { background: '#1e3a5f', border: 'none', borderRadius: '12px', padding: '0 20px', color: '#fff', fontSize: '20px', cursor: 'pointer', transition: 'background 0.2s' },
  sendBtnDisabled: { background: '#1a1a1a', cursor: 'not-allowed', color: '#444' },
  copyBtn: { background: 'transparent', border: '1px solid #2a2a2a', borderRadius: '6px', padding: '2px 8px', color: '#777', fontSize: '11px', cursor: 'pointer' },
  clearBtn: { float: 'right', background: 'transparent', border: '1px solid #2a2a2a', borderRadius: '8px', padding: '4px 10px', color: '#888', fontSize: '12px', cursor: 'pointer' },
  exampleBtn: { background: 'transparent', border: '1px solid #2a2a2a', borderRadius: '20px', padding: '6px 14px', color: '#666', fontSize: '13px', cursor: 'pointer', transition: 'all 0.2s' },
}

const EXAMPLE_QUERIES = [
  "What are the key findings on attention mechanisms in transformers?",
  "How many cs.LG papers were published per month in 2017?",
  "How do GANs suffer from mode collapse, and what fixes were proposed?",
  "Who are Yoshua Bengio's most frequent co-authors?",
]

function Citations({ citations }) {
  if (!citations || citations.length === 0) return null
  return (
    <div style={styles.citations}>
      <div style={styles.citationTitle}>Sources ({citations.length})</div>
      {citations.slice(0, 5).map((c, i) => (
        <div key={i} style={styles.citationItem}>
          <strong style={{ color: '#ccc' }}>{c.title || c.arxiv_id}</strong>
          {c.authors && c.authors.length > 0 && (
            <span style={{ color: '#666', marginLeft: '6px' }}>— {c.authors.slice(0, 2).join(', ')}{c.authors.length > 2 ? ' et al.' : ''}</span>
          )}
          {c.arxiv_id && <a href={`https://arxiv.org/abs/${c.arxiv_id}`} target="_blank" rel="noopener noreferrer" style={{ color: '#4a6ea8', marginLeft: '6px', fontSize: '11px', textDecoration: 'none' }}>[{c.arxiv_id}]</a>}
        </div>
      ))}
    </div>
  )
}

function formatCell(value) {
  if (value === null || value === undefined) return '—'
  if (Array.isArray(value)) return value.join(', ')
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 3 })
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function SqlResults({ results }) {
  if (!results || results.length === 0) return null
  const rows = results.slice(0, 50)
  const columns = [...new Set(rows.flatMap(r => Object.keys(r)))]
  return (
    <div style={styles.sqlResults}>
      <div style={{ ...styles.citationTitle, padding: '10px 12px 4px', color: '#4a6ea8' }}>
        Data ({results.length} row{results.length === 1 ? '' : 's'}{results.length > rows.length ? `, showing ${rows.length}` : ''})
      </div>
      <div className="table-wrap">
        <table className="data-table">
          <thead><tr>{columns.map(c => <th key={c}>{c.replace(/_/g, ' ')}</th>)}</tr></thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i}>
                {columns.map(c => <td key={c} className={typeof row[c] === 'number' ? 'num' : undefined}>{formatCell(row[c])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function StatusTrail({ steps, active }) {
  if (!steps.length && !active) return null
  return (
    <div style={{ ...styles.loadingDots, display: 'flex', flexDirection: 'column', gap: '2px', minWidth: '260px' }}>
      {steps.map((s, i) => <div key={i} className="status-line done fade-in">{s}</div>)}
      {active && <div className="status-line active">{steps.length ? 'Working…' : 'Planning…'}</div>}
    </div>
  )
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
  const [initial] = useState(loadHistory)
  const [messages, setMessages] = useState(initial.messages)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(initial.sessionId)
  const [slowStart, setSlowStart] = useState(false)
  const [steps, setSteps] = useState([])
  const bottomRef = useRef(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify({ messages: messages.slice(-HISTORY_LIMIT), sessionId }))
    } catch { /* storage full or disabled — history is best-effort */ }
  }, [messages, sessionId])

  function clearChat() {
    abortRef.current?.abort()
    setMessages([])
    setSessionId(null)
  }

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading, steps])

  async function sendMessage(query) {
    if (!query.trim() || loading) return

    const userMsg = { role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    // Show cold-start warning after 3s
    const slowTimer = setTimeout(() => setSlowStart(true), 3000)

    const controller = new AbortController()
    abortRef.current = controller

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        signal: controller.signal,
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream', ...(await authHeader()) },
        body: JSON.stringify({ query, session_id: sessionId }),
      })
      clearTimeout(slowTimer)
      setSlowStart(false)

      if (!res.ok) {
        const detail = await describeHttpError(res)
        setMessages(prev => [...prev, {
          role: 'assistant', content: detail,
          citations: [], sqlResults: null, provider: 'error',
        }])
        return
      }

      let answered = false
      await readSSE(res, (event, data) => {
        if (event === 'status') {
          setSteps(prev => [...prev, data.message])
        } else if (event === 'answer') {
          answered = true
          if (data.session_id) setSessionId(data.session_id)
          setMessages(prev => [...prev, {
            role: 'assistant',
            content: data.answer,
            citations: data.citations,
            sqlResults: data.sql_results,
            provider: data.provider,
          }])
        }
      })
      if (!answered) throw new Error('stream ended before an answer arrived')
    } catch (err) {
      clearTimeout(slowTimer)
      setSlowStart(false)
      const stopped = err.name === 'AbortError'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: stopped
          ? '*Stopped.*'
          : `Error: ${err.message}. The API may be waking up — please try again in a moment.`,
        citations: [],
        sqlResults: null,
        provider: 'error',
      }])
    } finally {
      abortRef.current = null
      setLoading(false)
      setSteps([])
      setTimeout(() => inputRef.current?.focus(), 100)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>
          Research Intelligence Agent
          <span style={styles.providerBadge}>LangGraph + pgvector</span>
          {signOut && (
            <button style={styles.clearBtn} onClick={() => { clearChat(); signOut() }} aria-label="Sign out" title={user?.email || ''}>Sign out</button>
          )}
          {messages.length > 0 && (
            <button style={{ ...styles.clearBtn, marginRight: '8px' }} onClick={clearChat} aria-label="Start a new chat">New chat</button>
          )}
        </div>
        <div style={styles.subtitle}>Ask about 50k ArXiv ML papers (2007–2018) — semantic search, SQL analytics, co-author graph</div>
      </div>

      <div style={styles.messages} role="log" aria-live="polite">
        {messages.length === 0 && (
          <div style={{ padding: '40px 0', textAlign: 'center' }}>
            <div style={{ color: '#444', marginBottom: '20px', fontSize: '15px' }}>Try an example:</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', justifyContent: 'center' }}>
              {EXAMPLE_QUERIES.map((q, i) => (
                <button key={i} style={styles.exampleBtn} onClick={() => sendMessage(q)}>{q}</button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className="fade-in" style={msg.role === 'user' ? styles.userMsg : styles.assistantMsg}>
            {msg.role === 'user' ? msg.content : (
              <div className="md">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: props => <a {...props} target="_blank" rel="noopener noreferrer" />,
                    table: props => <div className="table-wrap"><table {...props} /></div>,
                  }}
                >
                  {msg.content || ''}
                </ReactMarkdown>
              </div>
            )}
            {msg.role === 'assistant' && (
              <>
                <Citations citations={msg.citations} />
                <SqlResults results={msg.sqlResults} />
                <div style={{ marginTop: '8px', display: 'flex', alignItems: 'center', gap: '10px' }}>
                  {msg.provider && msg.provider !== 'stub' && msg.provider !== 'error' && (
                    <span style={{ fontSize: '11px', color: '#666' }}>via {msg.provider}</span>
                  )}
                  {msg.provider !== 'error' && msg.content && <CopyButton text={msg.content} />}
                </div>
              </>
            )}
          </div>
        ))}

        {slowStart && (
          <div style={styles.coldStartBanner}>
            API is waking up from sleep — this may take 10-30s on first request...
          </div>
        )}
        {loading && <StatusTrail steps={steps} active />}
        <div ref={bottomRef} />
      </div>

      <div style={styles.inputArea}>
        <textarea
          ref={inputRef}
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about ML papers... (Enter to send, Shift+Enter for newline)"
          rows={1}
          disabled={loading}
          aria-label="Your question"
          maxLength={4000}
        />
        {loading ? (
          <button style={styles.sendBtn} onClick={() => abortRef.current?.abort()} aria-label="Stop generating">■</button>
        ) : (
          <button
            style={!input.trim() ? { ...styles.sendBtn, ...styles.sendBtnDisabled } : styles.sendBtn}
            onClick={() => sendMessage(input)}
            disabled={!input.trim()}
            aria-label="Send"
          >
            ↑
          </button>
        )}
      </div>
    </div>
  )
}
