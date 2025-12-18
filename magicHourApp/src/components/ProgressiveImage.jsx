import { useState } from 'react'

function ProgressiveImage({ src, alt, blurPreview, disableExpand = false }) {
  const [isLoaded, setIsLoaded] = useState(false)
  const [isExpanded, setIsExpanded] = useState(false)

  const handleClick = (e) => {
    if (disableExpand) return
    e.stopPropagation()
    setIsExpanded(true)
  }

  return (
    <>
      <div
        className={`progressive-image ${isLoaded ? '' : 'loading'}`}
        onDoubleClick={handleClick}
        style={{ cursor: disableExpand ? 'pointer' : 'zoom-in', position: 'relative' }}
      >
        {/* ChatGPT-style blur preview */}
        {blurPreview && !isLoaded && (
          <img
            src={blurPreview}
            alt="Loading preview"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              filter: 'blur(20px)',
              transform: 'scale(1.1)',
              objectFit: 'cover',
              borderRadius: 'var(--radius-md)'
            }}
          />
        )}

        {/* Shimmer fallback if no blur preview */}
        {!isLoaded && !blurPreview && (
          <div className="placeholder shimmer" style={{ minHeight: '200px', minWidth: '200px' }}>
          </div>
        )}

        {/* Full resolution image */}
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          style={{
            opacity: isLoaded ? 1 : 0,
            transition: 'opacity 0.5s ease',
            position: 'relative',
            display: 'block'
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
