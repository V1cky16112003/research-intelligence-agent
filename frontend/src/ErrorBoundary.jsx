import React from 'react'

// A render error (e.g. malformed markdown from the model) should not blank the page.
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { error: null }
  }

  static getDerivedStateFromError(error) {
    return { error }
  }

  componentDidCatch(error, info) {
    console.error('UI crashed:', error, info.componentStack)
  }

  reset = () => {
    // Saved history may be what crashed the render; drop it so reload can recover.
    try { localStorage.removeItem('research-agent-history-v1') } catch { /* ignore */ }
    window.location.reload()
  }

  render() {
    if (!this.state.error) return this.props.children
    return (
      <div role="alert" style={{ maxWidth: 600, margin: '80px auto', padding: 24, color: '#ccc', textAlign: 'center' }}>
        <h2 style={{ color: '#fff' }}>Something went wrong</h2>
        <p>The page hit an unexpected error.</p>
        <button onClick={this.reset} style={{ marginTop: 12, padding: '8px 16px', borderRadius: 8, border: '1px solid #333', background: '#1e3a5f', color: '#fff', cursor: 'pointer' }}>
          Clear chat and reload
        </button>
      </div>
    )
  }
}
