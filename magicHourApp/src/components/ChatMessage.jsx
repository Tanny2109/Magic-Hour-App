import ProgressiveImage from './ProgressiveImage'
import ReasoningPanel from './ReasoningPanel'

function ChatMessage({ message, reasoning, selectedImage, onImageSelect }) {
  const { role, content, images, videos, description, isLoading, isGeneratingImage, reflection, enhancedPrompt, finalReasoning, id } = message

  const handleImageClick = (idx, url) => {
    if (onImageSelect) {
      onImageSelect({ msgId: id, imageIndex: idx, url })
    }
  }

  const isSelected = (idx) => {
    return selectedImage?.msgId === id && selectedImage?.imageIndex === idx
  }

  const isLive = isLoading || isGeneratingImage
  const reasoningSteps = (reasoning && reasoning.length) ? reasoning : (finalReasoning || [])
  const showReasoning = (reasoningSteps && reasoningSteps.length > 0) || isLive

  return (
    <div className={`message ${role}`}>
      <div className="message-avatar">
        {role === 'user' ? '👤' : '✨'}
      </div>
      <div className="message-content">
        {content && <div className="text-content">{content}</div>}

        {enhancedPrompt && (
          <div className="enhanced-prompt">
            <div className="enhanced-label">Enhanced Prompt</div>
            <div className="enhanced-text">{enhancedPrompt}</div>
          </div>
        )}

        {showReasoning && (
          <ReasoningPanel
            steps={reasoningSteps}
            isLive={isLive}
            autoExpandLive={true}
          />
        )}

        {isLoading && !images?.length && !videos?.length && (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        )}

        {isGeneratingImage && (
          <div className="progressive-image loading">
            <div className="placeholder">
              <div className="spinner"></div>
              <span>Generating images...</span>
            </div>
          </div>
        )}

        {/* Images in a horizontal row */}
        {images && images.length > 0 && (
          <div className={`image-grid ${images.length > 1 ? 'multi' : ''}`}>
            {images.map((url, idx) => (
              <div
                key={idx}
                className={`image-grid-item ${isSelected(idx) ? 'selected' : ''}`}
                onClick={() => handleImageClick(idx, url)}
              >
                <div className="image-index">V{idx + 1}</div>
                <ProgressiveImage
                  src={url}
                  alt={`Variation ${idx + 1}`}
                  disableExpand={true}
                />
                {isSelected(idx) && <div className="selected-badge">✓ Selected</div>}
              </div>
            ))}
          </div>
        )}

        {videos && videos.map((url, idx) => (
          <div key={idx} className="video-container">
            <video controls autoPlay loop muted>
              <source src={url} type="video/mp4" />
            </video>
          </div>
        ))}

        {description && (
          <div className="image-description">💬 {description}</div>
        )}

        {reflection && (
          <div className="agent-reflection">
            <div className="reflection-icon">✨</div>
            <div className="reflection-text">{reflection}</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
