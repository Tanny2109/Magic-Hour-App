import ProgressiveImage from './ProgressiveImage'
import ReasoningPanel from './ReasoningPanel'

function ChatMessage({ message, reasoning }) {
  const { role, content, images, videos, description, isLoading, isGeneratingImage } = message

  return (
    <div className={`message ${role}`}>
      <div className="message-avatar">
        {role === 'user' ? '👤' : '✨'}
      </div>
      <div className="message-content">
        {/* Text content */}
        {content && <div className="text-content">{content}</div>}
        
        {/* Loading state with reasoning */}
        {isLoading && !isGeneratingImage && !images?.length && (
          <>
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
            {reasoning && reasoning.length > 0 && (
              <ReasoningPanel steps={reasoning} isLive={true} />
            )}
          </>
        )}

        {/* Generating image state */}
        {isGeneratingImage && (
          <div className="progressive-image loading">
            <div className="placeholder">
              <div className="spinner"></div>
              <span>Generating image...</span>
            </div>
          </div>
        )}

        {/* Images */}
        {images && images.map((url, idx) => (
          <ProgressiveImage key={idx} src={url} alt={`Generated image ${idx + 1}`} />
        ))}

        {/* Videos */}
        {videos && videos.map((url, idx) => (
          <div key={idx} className="video-container">
            <video controls autoPlay loop muted>
              <source src={url} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          </div>
        ))}

        {/* Conversational description */}
        {description && (
          <div className="image-description">
            💬 {description}
          </div>
        )}
      </div>
    </div>
  )
}

export default ChatMessage
