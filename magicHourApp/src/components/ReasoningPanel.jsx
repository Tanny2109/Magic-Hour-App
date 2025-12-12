import { useState } from 'react'

function ReasoningPanel({ steps, isLive }) {
  const [isExpanded, setIsExpanded] = useState(true)

  if (!steps || steps.length === 0) return null

  return (
    <div className="reasoning-panel">
      <div 
        className={`reasoning-header ${isExpanded ? 'expanded' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="icon">▶</span>
        <span>🧠 Agent Reasoning</span>
        {isLive && <span style={{ color: 'var(--accent-primary)', fontSize: '0.75rem' }}>● Live</span>}
      </div>
      
      {isExpanded && (
        <div className="reasoning-content">
          {steps.map((step, idx) => (
            <div key={idx} className={`reasoning-step ${step.type === 'tool' ? 'tool-call' : ''}`}>
              <span className="step-icon">
                {step.type === 'tool' ? '🔧' : '💭'}
              </span>
              <div className="step-content">
                {step.type === 'tool' ? (
                  <>
                    <strong>Using: {step.name}</strong>
                    {step.args && (
                      <pre style={{ 
                        fontSize: '0.8rem', 
                        marginTop: '6px', 
                        opacity: 0.7,
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word'
                      }}>
                        {typeof step.args === 'string' ? step.args : JSON.stringify(step.args, null, 2)}
                      </pre>
                    )}
                  </>
                ) : (
                  <span>{step.content}</span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ReasoningPanel
