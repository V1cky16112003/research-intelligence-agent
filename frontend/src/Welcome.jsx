import React from 'react'
import './chat.css'

const EXAMPLES = [
  { tag: 'Semantic search', q: "What are the key findings on attention mechanisms in transformers?" },
  { tag: 'SQL analytics', q: "How many cs.LG papers were published per month in 2017?" },
  { tag: 'Semantic search', q: "How do GANs suffer from mode collapse, and what fixes were proposed?" },
  { tag: 'Co-author graph', q: "Who are Yoshua Bengio's most frequent co-authors?" },
]

export default function Welcome({ onPick, greetingName }) {
  return (
    <div className="welcome">
      <h1 className="welcome-title">{greetingName ? `Welcome back, ${greetingName}` : 'What would you like to research?'}</h1>
      <p className="welcome-scope">50k arXiv ML papers (2007–2018) — semantic search, SQL analytics, co-author graph</p>
      <div className="welcome-grid">
        {EXAMPLES.map(({ tag, q }) => (
          <button key={q} className="welcome-card" onClick={() => onPick(q)}>
            <span className="welcome-tag">{tag}</span>
            <span className="welcome-q">{q}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
