import React, { useEffect, useRef, useState } from 'react'
import UsageMeter from './UsageMeter.jsx'
import './sidebar.css'

const MOBILE_QUERY = '(max-width: 767px)'

function useIsMobile() {
  const [mobile, setMobile] = useState(() => !!window.matchMedia?.(MOBILE_QUERY).matches)
  useEffect(() => {
    const mq = window.matchMedia?.(MOBILE_QUERY)
    if (!mq) return
    const on = () => setMobile(mq.matches)
    mq.addEventListener('change', on)
    return () => mq.removeEventListener('change', on)
  }, [])
  return mobile
}

const Svg = ({ children, size = 16, fill = 'none' }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill={fill} stroke="currentColor" strokeWidth="1.8"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">{children}</svg>
)
const IconPlus = () => <Svg><path d="M12 5v14M5 12h14" /></Svg>
const IconMenu = () => <Svg size={18}><path d="M4 6h16M4 12h16M4 18h16" /></Svg>
const IconPanel = () => <Svg><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M9 4v16" /></Svg>
const IconSearch = () => <Svg size={14}><circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" /></Svg>
const IconPin = ({ filled }) => (
  <Svg fill={filled ? 'currentColor' : 'none'}><path d="M9 4h6l-1 6 3 3H7l3-3-1-6zM12 13v7" /></Svg>
)
const IconEdit = () => <Svg><path d="M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4" /></Svg>
const IconTrash = () => <Svg><path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3" /></Svg>

// Hamburger for the parent to render in the main header when the sidebar is collapsed / on mobile.
export function SidebarToggle({ onClick }) {
  return (
    <button type="button" className="sb-icon sb-hamburger" onClick={onClick} aria-label="Open sidebar">
      <IconMenu />
    </button>
  )
}

// Own component so the once-only guard resets each time an edit starts (Enter/Esc then blur must not double-fire).
function RenameInput({ item, finishEdit }) {
  const done = useRef(false)
  const finish = (save, value) => {
    if (done.current) return
    done.current = true
    finishEdit(item, save, value)
  }
  return (
    <li className="sb-row">
      <input
        className="sb-rename"
        aria-label="Rename chat"
        defaultValue={item.title}
        autoFocus
        maxLength={80}
        onFocus={e => e.target.select()}
        onKeyDown={e => {
          if (e.key === 'Enter') finish(true, e.currentTarget.value)
          else if (e.key === 'Escape') { e.stopPropagation(); finish(false) }
        }}
        onBlur={e => finish(true, e.currentTarget.value)}
      />
    </li>
  )
}

function Row({ item, active, editing, confirming, handlers }) {
  const { select, startEdit, finishEdit, askDelete, cancelDelete, confirmDelete, onPin } = handlers

  if (editing) return <RenameInput item={item} finishEdit={finishEdit} />

  if (confirming) {
    return (
      <li className="sb-row sb-confirm" data-active={active || undefined}>
        <span className="sb-confirm-q">Delete?</span>
        <button type="button" className="sb-btn sb-btn-danger" onClick={() => confirmDelete(item.id)}>Yes</button>
        <button type="button" className="sb-btn" onClick={cancelDelete} autoFocus>No</button>
      </li>
    )
  }

  return (
    <li className="sb-row" data-active={active || undefined}>
      <button
        type="button"
        className="sb-title"
        aria-current={active ? 'true' : undefined}
        title={item.title}
        onClick={() => select(item.id)}
        onDoubleClick={() => startEdit(item.id)}
      >
        {item.title}
      </button>
      <span className="sb-actions">
        <button type="button" className="sb-icon" aria-label={item.pinned ? 'Unpin chat' : 'Pin chat'}
          aria-pressed={!!item.pinned} onClick={() => onPin?.(item.id)}>
          <IconPin filled={item.pinned} />
        </button>
        <button type="button" className="sb-icon" aria-label="Rename chat" onClick={() => startEdit(item.id)}>
          <IconEdit />
        </button>
        <button type="button" className="sb-icon" aria-label="Delete chat" onClick={() => askDelete(item.id)}>
          <IconTrash />
        </button>
      </span>
    </li>
  )
}

export default function Sidebar({
  groups = [], activeId, query = '', onQuery, onSelect, onNew, onRename, onDelete, onPin,
  collapsed, onToggle, user, onSignOut, usage,
}) {
  const mobile = useIsMobile()
  const searchRef = useRef(null)
  const [editingId, setEditingId] = useState(null)
  const [confirmId, setConfirmId] = useState(null)
  const open = !collapsed
  const email = typeof user === 'string' ? user : user?.email

  useEffect(() => {
    const onKey = e => {
      if ((e.metaKey || e.ctrlKey) && !e.shiftKey && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        searchRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (!mobile || !open) return
    const onEsc = e => { if (e.key === 'Escape') onToggle?.() }
    window.addEventListener('keydown', onEsc)
    return () => window.removeEventListener('keydown', onEsc)
  }, [mobile, open, onToggle])

  const closeIfMobile = () => { if (mobile) onToggle?.() }
  const handlers = {
    select: id => { onSelect?.(id); closeIfMobile() },
    startEdit: id => { setConfirmId(null); setEditingId(id) },
    finishEdit: (item, save, value) => {
      setEditingId(null)
      const title = (value || '').trim()
      if (save && title && title !== item.title) onRename?.(item.id, title)
    },
    askDelete: id => { setEditingId(null); setConfirmId(id) },
    cancelDelete: () => setConfirmId(null),
    confirmDelete: id => { setConfirmId(null); onDelete?.(id) },
    onPin,
  }

  const visible = groups.filter(g => g.items?.length)

  return (
    <>
      {mobile && open && <div className="sb-backdrop" onClick={onToggle} aria-hidden="true" />}
      <aside className="sb" data-collapsed={collapsed || undefined} data-mobile={mobile || undefined} aria-label="Chat history">
        <div className="sb-inner">
          <div className="sb-head">
            <span className="sb-brand">
              <span className="sb-mark" aria-hidden="true">R</span>
              <span className="sb-name">Research Agent</span>
            </span>
            <button type="button" className="sb-icon" onClick={onToggle}
              aria-label={mobile ? 'Close sidebar' : 'Collapse sidebar'}>
              <IconPanel />
            </button>
          </div>

          <div className="sb-top">
            <button type="button" className="sb-new" aria-label="Start a new chat"
              onClick={() => { onNew?.(); closeIfMobile() }}>
              <IconPlus /> New chat
            </button>
            <label className="sb-search">
              <IconSearch />
              <input ref={searchRef} type="search" aria-label="Search chats" placeholder="Search chats"
                value={query} onChange={e => onQuery?.(e.target.value)} />
              <kbd aria-hidden="true">⌘K</kbd>
            </label>
          </div>

          <nav className="sb-list" aria-label="Conversations">
            {visible.length === 0 && (
              <p className="sb-empty">{query.trim() ? `No chats match “${query.trim()}”` : 'No chats yet'}</p>
            )}
            {visible.map(g => (
              <section key={g.label} className="sb-group">
                <h3 className="sb-label">{g.label}</h3>
                <ul>
                  {g.items.map(item => (
                    <Row key={item.id} item={item} active={item.id === activeId}
                      editing={editingId === item.id} confirming={confirmId === item.id} handlers={handlers} />
                  ))}
                </ul>
              </section>
            ))}
          </nav>

          <div className="sb-foot">
            <UsageMeter usage={usage} />
            {(email || onSignOut) && (
              <div className="sb-account">
                <span className="sb-email" title={email}>{email}</span>
                {onSignOut && (
                  <button type="button" className="sb-btn" aria-label="Sign out" onClick={onSignOut}>Sign out</button>
                )}
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  )
}
