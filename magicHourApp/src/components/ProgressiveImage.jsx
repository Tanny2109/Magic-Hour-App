import { useState } from 'react'

function ProgressiveImage({ src, alt, disableExpand = false }) {
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
        className={`progressive-image ${isLoaded ? 'loaded' : 'loading'}`}
        onDoubleClick={handleClick}
        style={{ cursor: disableExpand ? 'pointer' : 'zoom-in' }}
      >
        <img
          src={src}
          alt={alt}
          onLoad={() => setIsLoaded(true)}
          className={isLoaded ? 'loaded' : ''}
        />
      </div>

      {/* Lightbox for expanded view */}
      {isExpanded && (
        <div className="lightbox" onClick={() => setIsExpanded(false)}>
          <img src={src} alt={alt} />
          <div className="lightbox-close">✕</div>
        </div>
      )}
    </>
  )
}

export default ProgressiveImage
