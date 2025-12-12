import { useState } from 'react'

function ProgressiveImage({ src, alt }) {
  const [isLoaded, setIsLoaded] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <>
      <div 
        className={`progressive-image ${isLoaded ? '' : 'loading'}`}
        onClick={() => setIsExpanded(true)}
        style={{ cursor: 'pointer' }}
      >
        {!isLoaded && (
          <div className="placeholder shimmer" style={{ minHeight: '200px', minWidth: '200px' }}>
          </div>
        )}
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          style={{ 
            opacity: isLoaded ? 1 : 0,
            transition: 'opacity 0.5s ease'
          }}
        />
      </div>

      {/* Lightbox for expanded view */}
      {isExpanded && (
        <div 
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.9)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            cursor: 'pointer',
            animation: 'fadeIn 0.2s ease'
          }}
          onClick={() => setIsExpanded(false)}
        >
          <img
            src={src}
            alt={alt}
            style={{
              maxWidth: '90vw',
              maxHeight: '90vh',
              borderRadius: 'var(--radius-lg)',
              boxShadow: 'var(--shadow-lg)'
            }}
          />
          <div style={{
            position: 'absolute',
            top: '20px',
            right: '20px',
            color: 'white',
            fontSize: '1.5rem',
            opacity: 0.7
          }}>
            ✕
          </div>
        </div>
      )}
    </>
  )
}

export default ProgressiveImage
