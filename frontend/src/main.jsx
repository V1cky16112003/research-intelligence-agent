import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import ErrorBoundary from './ErrorBoundary.jsx'
import AuthGate from './AuthGate.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ErrorBoundary>
      <AuthGate>{auth => <App {...auth} />}</AuthGate>
    </ErrorBoundary>
  </React.StrictMode>
)
