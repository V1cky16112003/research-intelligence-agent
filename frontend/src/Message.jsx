import React, { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import './markdown.css'
import './chat.css'

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      className="msg-copy"
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

function Citations({ citations }) {
  if (!citations || citations.length === 0) return null
  return (
    <section className="sources" aria-label="Sources">
      <h3 className="panel-title">Sources ({citations.length})</h3>
      <ol className="source-list">
        {citations.slice(0, 5).map((c, i) => (
          <li key={i} className="source-card">
            <span className="source-num">{i + 1}</span>
            <div className="source-body">
              <div className="source-title">{c.title || c.arxiv_id}</div>
              <div className="source-meta">
                {c.arxiv_id && (
                  <a className="source-id" href={`https://arxiv.org/abs/${c.arxiv_id}`} target="_blank" rel="noopener noreferrer">{c.arxiv_id}</a>
                )}
                {c.authors && c.authors.length > 0 && (
                  <span>{c.authors.slice(0, 2).join(', ')}{c.authors.length > 2 ? ' et al.' : ''}</span>
                )}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
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
    <section className="sql-results">
      <h3 className="panel-title">
        Data ({results.length} row{results.length === 1 ? '' : 's'}{results.length > rows.length ? `, showing ${rows.length}` : ''})
      </h3>
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
    </section>
  )
}

export default function Message({ msg }) {
  if (msg.role === 'user') {
    return <div className="msg msg-user fade-in">{msg.content}</div>
  }
  const isError = msg.provider === 'error'
  const showProvider = msg.provider && msg.provider !== 'stub' && !isError
  return (
    <div className={`msg msg-assistant fade-in${isError ? ' msg-error' : ''}`}>
      <div className="md">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" />,
            table: ({ node, ...props }) => <div className="table-wrap"><table {...props} /></div>,
          }}
        >
          {msg.content || ''}
        </ReactMarkdown>
      </div>
      <Citations citations={msg.citations} />
      <SqlResults results={msg.sqlResults} />
      <div className="msg-foot">
        {showProvider && <span className="provider-chip">via {msg.provider}</span>}
        {!isError && msg.content && <CopyButton text={msg.content} />}
      </div>
    </div>
  )
}
