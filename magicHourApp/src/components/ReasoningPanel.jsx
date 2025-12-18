import { useState } from 'react'

function ReasoningPanel({ steps, isLive }) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!steps || steps.length === 0) return null

  // Get the latest step to show when collapsed
  const latestStep = steps[steps.length - 1]

  return (
    <div className="reasoning-panel">
      <div
        className={`reasoning-header ${isExpanded ? 'expanded' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <svg
          className={`expand-arrow ${isExpanded ? 'expanded' : ''}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="none"
        >
          <path
            d="M3 4.5L6 7.5L9 4.5"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        <span className="reasoning-label">
          {isExpanded ? '🧠 Agent Reasoning' : '🧠 Thinking'}
        </span>
        {isLive && !isExpanded && <span className="live-indicator">●</span>}
        {steps.length > 1 && !isExpanded && (
          <span className="step-count">{steps.length} steps</span>
        )}
      </div>

      {isExpanded ? (
        <div className="reasoning-content expanded">
          {steps.map((step, idx) => (
            <div key={idx} className={`reasoning-step ${step.type === 'tool' ? 'tool-call' : ''}`}>
              <span className="step-icon">
                {step.type === 'tool' ? '🔧' : '💭'}
              </span>
              <div className="step-content">
                {step.type === 'tool' ? (
                  <strong>{step.name}</strong>
                ) : (
                  <span style={{ whiteSpace: 'pre-wrap' }}>{step.content}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="reasoning-content collapsed">
          <div className={`reasoning-step current ${latestStep.type === 'tool' ? 'tool-call' : ''}`}>
            <span className="step-icon">
              {latestStep.type === 'tool' ? '🔧' : '💭'}
            </span>
            <div className="step-content">
              {latestStep.type === 'tool' ? (
                <strong>{latestStep.name}</strong>
              ) : (
                <span style={{ whiteSpace: 'pre-wrap' }}>{latestStep.content}</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ReasoningPanel
