import React, { useState, useRef, useEffect } from 'react'

const API_URL = import.meta.env.VITE_API_URL || ''

// --- Styles (inline for simplicity, no CSS file needed) ---
const styles = {
  container: { display: 'flex', flexDirection: 'column', height: '100vh', maxWidth: '900px', margin: '0 auto', padding: '0 16px' },
  header: { padding: '20px 0 12px', borderBottom: '1px solid #2a2a2a' },
  title: { fontSize: '20px', fontWeight: 600, color: '#fff' },
  subtitle: { fontSize: '13px', color: '#666', marginTop: '4px' },
  providerBadge: { display: 'inline-block', fontSize: '11px', padding: '2px 8px', borderRadius: '12px', marginLeft: '8px', background: '#1a1a2e', color: '#6c8ebf' },
  messages: { flex: 1, overflowY: 'auto', padding: '20px 0', display: 'flex', flexDirection: 'column', gap: '16px' },
  userMsg: { alignSelf: 'flex-end', background: '#1e3a5f', padding: '12px 16px', borderRadius: '16px 16px 4px 16px', maxWidth: '75%', fontSize: '15px', lineHeight: '1.5' },
  assistantMsg: { alignSelf: 'flex-start', background: '#1a1a1a', padding: '12px 16px', borderRadius: '16px 16px 16px 4px', maxWidth: '85%', fontSize: '15px', lineHeight: '1.6', border: '1px solid #2a2a2a' },
  citations: { marginTop: '12px', padding: '10px', background: '#111', borderRadius: '8px', fontSize: '13px', border: '1px solid #222' },
  citationTitle: { color: '#888', marginBottom: '6px', fontSize: '12px', textTransform: 'uppercase', letterSpacing: '0.05em' },
  citationItem: { padding: '6px 0', borderBottom: '1px solid #1a1a1a', color: '#aaa' },
  sqlResults: { marginTop: '10px', padding: '10px', background: '#0a1628', borderRadius: '8px', fontSize: '12px', color: '#6c8ebf', border: '1px solid #1a2a4a', fontFamily: 'monospace', maxHeight: '200px', overflowY: 'auto' },
  loadingDots: { alignSelf: 'flex-start', padding: '12px 16px', background: '#1a1a1a', borderRadius: '16px', border: '1px solid #2a2a2a' },
  coldStartBanner: { background: '#1a1a00', border: '1px solid #333300', borderRadius: '8px', padding: '10px 14px', fontSize: '13px', color: '#aaaa00', marginBottom: '12px', textAlign: 'center' },
  inputArea: { padding: '16px 0 24px', borderTop: '1px solid #2a2a2a', display: 'flex', gap: '8px' },
  input: { flex: 1, background: '#1a1a1a', border: '1px solid #2a2a2a', borderRadius: '12px', padding: '12px 16px', color: '#e8e8e8', fontSize: '15px', outline: 'none', resize: 'none' },
  sendBtn: { background: '#1e3a5f', border: 'none', borderRadius: '12px', padding: '0 20px', color: '#fff', fontSize: '20px', cursor: 'pointer', transition: 'background 0.2s' },
  sendBtnDisabled: { background: '#1a1a1a', cursor: 'not-allowed', color: '#444' },
  exampleBtn: { background: 'transparent', border: '1px solid #2a2a2a', borderRadius: '20px', padding: '6px 14px', color: '#666', fontSize: '13px', cursor: 'pointer', transition: 'all 0.2s' },
}

const EXAMPLE_QUERIES = [
  "What are the key findings on attention mechanisms in transformers?",
  "How many cs.LG papers were published per month in 2023?",
  "Summarize recent advances in diffusion models for image generation",
  "Compare RAG vs fine-tuning approaches for LLM knowledge updates",
]

function LoadingDots() {
  return (
    <div style={styles.loadingDots}>
      <span style={{ animation: 'pulse 1.4s ease-in-out infinite', display: 'inline-block' }}>●</span>
      <span style={{ animation: 'pulse 1.4s ease-in-out 0.2s infinite', display: 'inline-block', margin: '0 4px' }}>●</span>
      <span style={{ animation: 'pulse 1.4s ease-in-out 0.4s infinite', display: 'inline-block' }}>●</span>
      <style>{`@keyframes pulse { 0%,80%,100%{opacity:.2} 40%{opacity:1} }`}</style>
    </div>
  )
}

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
          {c.arxiv_id && <span style={{ color: '#4a6ea8', marginLeft: '6px', fontSize: '11px' }}>[{c.arxiv_id}]</span>}
        </div>
      ))}
    </div>
  )
}

function SqlResults({ results }) {
  if (!results || results.length === 0) return null
  return (
    <div style={styles.sqlResults}>
      <div style={{ color: '#4a6ea8', marginBottom: '6px', fontSize: '11px' }}>SQL ANALYTICS RESULTS</div>
      <pre style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>{JSON.stringify(results.slice(0, 10), null, 2)}</pre>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(null)
  const [slowStart, setSlowStart] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  async function sendMessage(query) {
    if (!query.trim() || loading) return

    const userMsg = { role: 'user', content: query }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    // Show cold-start warning after 3s
    const slowTimer = setTimeout(() => setSlowStart(true), 3000)

    try {
      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, session_id: sessionId }),
      })
      clearTimeout(slowTimer)
      setSlowStart(false)

      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()

      if (data.session_id) setSessionId(data.session_id)

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.answer,
        citations: data.citations,
        sqlResults: data.sql_results,
        provider: data.provider,
      }])
    } catch (err) {
      clearTimeout(slowTimer)
      setSlowStart(false)
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Error: ${err.message}. The API may be waking up — please try again in a moment.`,
        citations: [],
        sqlResults: null,
        provider: 'error',
      }])
    } finally {
      setLoading(false)
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
        </div>
        <div style={styles.subtitle}>Ask questions about ArXiv ML papers — semantic search + SQL analytics</div>
      </div>

      <div style={styles.messages}>
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
          <div key={i} style={msg.role === 'user' ? styles.userMsg : styles.assistantMsg}>
            {msg.content}
            {msg.role === 'assistant' && (
              <>
                <Citations citations={msg.citations} />
                <SqlResults results={msg.sqlResults} />
                {msg.provider && msg.provider !== 'stub' && msg.provider !== 'error' && (
                  <div style={{ marginTop: '8px', fontSize: '11px', color: '#444' }}>via {msg.provider}</div>
                )}
              </>
            )}
          </div>
        ))}

        {slowStart && (
          <div style={styles.coldStartBanner}>
            API is waking up from sleep — this may take 10-30s on first request...
          </div>
        )}
        {loading && <LoadingDots />}
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
        />
        <button
          style={loading || !input.trim() ? { ...styles.sendBtn, ...styles.sendBtnDisabled } : styles.sendBtn}
          onClick={() => sendMessage(input)}
          disabled={loading || !input.trim()}
        >
          ↑
        </button>
      </div>
    </div>
  )
}
