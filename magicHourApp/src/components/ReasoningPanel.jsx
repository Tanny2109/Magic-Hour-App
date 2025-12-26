import { useState, useEffect, useRef } from 'react'

function ReasoningPanel({ steps, isLive, autoExpandLive = true }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const contentRef = useRef(null)

  useEffect(() => {
    if (autoExpandLive && isLive) {
      setIsExpanded(true)
    }
  }, [autoExpandLive, isLive])

  // Auto-scroll content when expanded and new steps arrive
  useEffect(() => {
    if (isExpanded && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [steps, isExpanded])

  if ((!steps || steps.length === 0) && !isLive) return null

  const lastStep = steps && steps.length ? steps[steps.length - 1] : null
  const previewText = lastStep?.content || ''
  const previewTrimmed = previewText.length > 140
    ? `${previewText.slice(0, 140)}...`
    : previewText

  const getStepIcon = (step) => {
    switch (step?.type) {
      case 'tool': return '🔧'
      case 'visual': return '👁️'
      case 'reasoning': return '💭'
      case 'thought': return '🧠'
      default: return '💭'
    }
  }

  const getStepLabel = (step) => {
    switch (step?.type) {
      case 'tool': return 'Calling tool'
      case 'visual': return 'Analyzing images'
      case 'reasoning': return 'Thinking'
      case 'thought': return 'Processing'
      default: return 'Thinking'
    }
  }

  const handleToggle = () => setIsExpanded(prev => !prev)
  const handleKeyDown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      handleToggle()
    }
  }

  return (
    <div className={`reasoning-panel ${isLive ? 'live' : ''}`}>
      {isExpanded ? (
        <div className="reasoning-expanded">
          <div
            className="reasoning-header"
            role="button"
            tabIndex={0}
            onClick={handleToggle}
            onKeyDown={handleKeyDown}
          >
            <span className="reasoning-title">🧠 Agent Reasoning</span>
            <span className="reasoning-collapse">
              Collapse
              <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
                <path d="M5.3 12.7a1 1 0 0 1 0-1.4l4-4a1 1 0 0 1 1.4 0l4 4a1 1 0 1 1-1.4 1.4L10 9.4l-3.3 3.3a1 1 0 0 1-1.4 0z" />
              </svg>
            </span>
          </div>
          <div className="reasoning-steps" ref={contentRef}>
            {steps && steps.map((step, idx) => (
              <div
                key={idx}
                className={`reasoning-step ${step.type || ''}`}
                style={{ animationDelay: `${idx * 0.05}s` }}
              >
                <div className="step-header">
                  <span className="step-icon">{getStepIcon(step)}</span>
                  <span className="step-label">{getStepLabel(step)}</span>
                  {step.type === 'tool' && step.name && (
                    <span className="tool-name">{step.name}</span>
                  )}
                </div>
                {step.content && (
                  <div className="step-content">{step.content}</div>
                )}
              </div>
            ))}
            {isLive && (
              <div className="reasoning-step thinking">
                <div className="step-header">
                  <span className="step-icon">⏳</span>
                  <span className="step-label">Still thinking</span>
                  <span className="thinking-dots"><span>.</span><span>.</span><span>.</span></span>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div
          className="reasoning-collapsed"
          role="button"
          tabIndex={0}
          onClick={handleToggle}
          onKeyDown={handleKeyDown}
        >
          <div className="reasoning-line">
            <span className="reasoning-icon">🧠</span>
            <span className="reasoning-label">Agent reasoning</span>
            {isLive && (
              <span className="thinking-dots"><span>.</span><span>.</span><span>.</span></span>
            )}
          </div>
          <div className="reasoning-preview">
            {previewTrimmed || 'Awaiting reasoning...'}
            {isLive && <span className="cursor">|</span>}
          </div>
          <div className="reasoning-expand-hint">
            <span className="step-count">{steps ? steps.length : 0} steps</span>
            <span className="expand-text">Click to expand</span>
            <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path d="M14.7 7.3a1 1 0 0 1 0 1.4l-4 4a1 1 0 0 1-1.4 0l-4-4a1 1 0 1 1 1.4-1.4L10 10.6l3.3-3.3a1 1 0 0 1 1.4 0z" />
            </svg>
          </div>
        </div>
      )}
    </div>
  )
}

export default ReasoningPanel
