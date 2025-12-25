import { useState, useEffect, useRef } from 'react'

function ReasoningPanel({ steps, isLive }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [displayedText, setDisplayedText] = useState('')
  const [currentStepIndex, setCurrentStepIndex] = useState(0)
  const contentRef = useRef(null)

  // Get the current step being displayed
  const currentStep = steps[currentStepIndex]
  const hasMultipleSteps = steps.length > 1

  // Animate text appearing character by character for the current step
  useEffect(() => {
    if (!currentStep || isExpanded) return

    const fullText = currentStep.content || ''
    let charIndex = 0
    setDisplayedText('')

    const interval = setInterval(() => {
      if (charIndex < fullText.length) {
        setDisplayedText(fullText.slice(0, charIndex + 1))
        charIndex++
      } else {
        clearInterval(interval)
        // Move to next step after a pause
        if (currentStepIndex < steps.length - 1) {
          setTimeout(() => {
            setCurrentStepIndex(prev => prev + 1)
          }, 500)
        }
      }
    }, 15) // Speed of typing

    return () => clearInterval(interval)
  }, [currentStep, currentStepIndex, isExpanded])

  // Reset when steps change significantly
  useEffect(() => {
    if (steps.length > 0 && currentStepIndex >= steps.length) {
      setCurrentStepIndex(steps.length - 1)
    }
  }, [steps.length, currentStepIndex])

  // Auto-scroll content when expanded
  useEffect(() => {
    if (isExpanded && contentRef.current) {
      contentRef.current.scrollTop = contentRef.current.scrollHeight
    }
  }, [steps, isExpanded])

  if (!steps || steps.length === 0) return null

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

  const formatContent = (content) => {
    if (!content) return ''
    // Truncate very long content in collapsed view
    if (content.length > 200) {
      return content.slice(0, 200) + '...'
    }
    return content
  }

  return (
    <div className={`reasoning-panel ${isExpanded ? 'expanded' : ''} ${isLive ? 'live' : ''}`}>
      {/* Collapsed view - shows current thinking line */}
      {!isExpanded && (
        <div className="reasoning-collapsed" onClick={() => setIsExpanded(true)}>
          <div className="reasoning-line">
            <span className="reasoning-icon">{getStepIcon(currentStep)}</span>
            <span className="reasoning-label">{getStepLabel(currentStep)}</span>
            {isLive && <span className="thinking-dots"><span>.</span><span>.</span><span>.</span></span>}
          </div>
          <div className="reasoning-preview">
            {formatContent(displayedText)}
            {isLive && displayedText.length < (currentStep?.content?.length || 0) && (
              <span className="cursor">|</span>
            )}
          </div>
          {hasMultipleSteps && (
            <div className="reasoning-expand-hint">
              <span className="step-count">{steps.length} steps</span>
              <span className="expand-text">Click to expand</span>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M3 4.5L6 7.5L9 4.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
          )}
        </div>
      )}

      {/* Expanded view - shows all steps */}
      {isExpanded && (
        <div className="reasoning-expanded">
          <div className="reasoning-header" onClick={() => setIsExpanded(false)}>
            <span className="reasoning-title">🧠 Agent Reasoning</span>
            <div className="reasoning-collapse">
              <span>Collapse</span>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
                <path d="M3 7.5L6 4.5L9 7.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              </svg>
            </div>
          </div>
          <div className="reasoning-steps" ref={contentRef}>
            {steps.map((step, idx) => (
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
      )}
    </div>
  )
}

export default ReasoningPanel
