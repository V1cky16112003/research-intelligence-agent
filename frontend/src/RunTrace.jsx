import React from 'react'
import './chat.css'

const NODES = ['Planner', 'Executor', 'Critic', 'Reporter']

// Status text comes from app/main.py::_describe_node, not a node field, so infer the node.
// Order matters: "Planned: answer from context" and "Refining the answer" both contain "answer".
const NODE_PATTERNS = [/plan/i, /gather|evidence|retriev|execut|search|query/i, /refin|check|critic|review/i, /draft|report|writ|answer/i]

function inferNode(step) {
  const i = NODE_PATTERNS.findIndex(re => re.test(step))
  return i === -1 ? null : i
}

export default function RunTrace({ steps = [], active = false }) {
  if (!steps.length && !active) return null

  const detail = {}
  const unknown = []
  let last = null
  for (const s of steps) {
    const n = inferNode(s)
    if (n === null) unknown.push(s)
    else { detail[n] = s; last = n }
  }
  // "Refining" means the critic sent the run back to the executor.
  const current = !active ? null : last === null ? 0 : /refin/i.test(steps[steps.length - 1]) ? 1 : Math.min(last + 1, 3)

  return (
    <div className="run-trace fade-in" role="status" aria-live="polite">
      {!steps.length ? (
        <div className="trace-planning">Planning…</div>
      ) : (
        <ol className="trace-steps">
          {NODES.map((name, i) => {
            const state = i === current ? 'active' : detail[i] ? 'done' : 'todo'
            return (
              <li key={name} className={`trace-step ${state}`} aria-current={state === 'active' ? 'step' : undefined}>
                <span className="trace-name">{name}</span>
                {detail[i] && <span className="trace-detail">{detail[i]}</span>}
              </li>
            )
          })}
        </ol>
      )}
      {unknown.map((s, i) => <div key={i} className="trace-line">{s}</div>)}
    </div>
  )
}
