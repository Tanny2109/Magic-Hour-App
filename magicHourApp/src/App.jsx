import { useState, useRef, useEffect } from 'react'
import './App.css'
import ChatMessage from './components/ChatMessage'

// API URL configuration
// In production (window.location.protocol !== 'http:' on localhost), use same origin
// In development, use localhost:8000
const getApiUrl = () => {
  // Check if VITE_API_URL is explicitly set
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL
  }

  // If running on localhost (development), use localhost:8000
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000'
  }

  // Otherwise (production), use the same origin as the frontend
  return window.location.origin
}

const API_URL = getApiUrl()

// Debug: Log API URL in production
if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
  console.log('🔗 API URL:', API_URL)
}

// Helper to convert relative URLs to absolute in development
const toAbsoluteUrl = (url) => {
  // If URL is already absolute (starts with http:// or https://), return as-is
  if (url.startsWith('http://') || url.startsWith('https://')) {
    return url
  }

  // If URL is relative and we're in development, prepend API_URL
  if (url.startsWith('/') && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')) {
    return `${API_URL}${url}`
  }

  // Otherwise (production with relative URL), return as-is (browser will resolve)
  return url
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentReasoning, setCurrentReasoning] = useState([])
  const [showSettings, setShowSettings] = useState(false)
  const [settings, setSettings] = useState({
    mode: 'fast',
    aspectRatio: 'square'
  })
  const [selectedImage, setSelectedImage] = useState(null) // {msgId, imageIndex, url}
  const [sessionId, setSessionId] = useState(null) // Session persistence for agent memory
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages, currentReasoning])

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setIsLoading(true)
    setCurrentReasoning([])

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])

    // Add assistant placeholder
    const assistantMsgId = Date.now()
    setMessages(prev => [...prev, {
      role: 'assistant',
      id: assistantMsgId,
      content: '',
      images: [],
      videos: [],
      description: '',
      reflection: '',
      enhancedPrompt: '',
      finalReasoning: [],
      isLoading: true
    }])

    try {
      // Include message history for context (especially image/video paths)
      const historyForContext = messages.map(msg => ({
        role: msg.role,
        content: msg.content,
        images: msg.images || [],
        videos: msg.videos || []
      }))

      // Include selected image reference if user has selected one
      let messageWithContext = userMessage
      if (selectedImage) {
        messageWithContext = `[User selected image V${selectedImage.imageIndex + 1}]\n${userMessage}`
      }

      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: messageWithContext,
          settings: { ...settings, selectedImage: selectedImage },
          history: historyForContext,
          session_id: sessionId  // CRITICAL: Send session_id for conversation memory
        })
      })

      // Clear selection after sending
      setSelectedImage(null)

      if (!response.ok) throw new Error('Failed to connect to server')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              handleSSEEvent(data, assistantMsgId)
            } catch (e) {
              console.error('Failed to parse SSE:', e)
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMsgId 
          ? { ...msg, content: `Error: ${error.message}. Make sure the backend is running.`, isLoading: false }
          : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleSSEEvent = (data, msgId) => {
    switch (data.type) {
      case 'reasoning':
        const reasoningStep = {
          type: 'thought',
          content: data.content,
          collapsible: false,
          timestamp: Date.now()
        }
        setCurrentReasoning(prev => [...prev, reasoningStep])
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, finalReasoning: [...(msg.finalReasoning || []), reasoningStep] }
            : msg
        ))
        break

      case 'reasoning_step':
        // Collapsible reasoning from agent
        const detailedReasoning = {
          type: 'reasoning',
          content: data.content,
          collapsible: data.collapsible !== false,
          timestamp: Date.now()
        }
        setCurrentReasoning(prev => [...prev, detailedReasoning])
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, finalReasoning: [...(msg.finalReasoning || []), detailedReasoning] }
            : msg
        ))
        break

      case 'visual_analysis':
        // Visual context analysis
        const visualAnalysis = {
          type: 'visual',
          content: data.content,
          collapsible: data.collapsible !== false,
          timestamp: Date.now()
        }
        setCurrentReasoning(prev => [...prev, visualAnalysis])
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, finalReasoning: [...(msg.finalReasoning || []), visualAnalysis] }
            : msg
        ))
        break

      case 'tool_call':
        const toolStep = {
          type: 'tool',
          name: data.tool,
          timestamp: Date.now()
        }
        setCurrentReasoning(prev => [...prev, toolStep])
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, finalReasoning: [...(msg.finalReasoning || []), toolStep] }
            : msg
        ))
        break

      case 'image_preview':
        // Add blur thumbnail for progressive loading
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? {
                ...msg,
                imagePreviews: [...(msg.imagePreviews || []), data.blur_data]
              }
            : msg
        ))
        break

      case 'image_progress':
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, isGeneratingImage: true }
            : msg
        ))
        break

      case 'image_complete':
        console.log('📸 Image complete:', data.url)
        setMessages(prev => {
          const updated = prev.map(msg => {
            if (msg.id === msgId) {
              const absoluteUrl = toAbsoluteUrl(data.url)
              const newMsg = {
                ...msg,
                images: [...(msg.images || []), absoluteUrl],
                isGeneratingImage: false,
                isLoading: false
              }
              console.log('✅ Updated message with image:', newMsg.images)
              return newMsg
            }
            return msg
          })
          console.log('📋 All messages:', updated)
          return updated
        })
        break

      case 'video_complete':
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? {
                ...msg,
                videos: [...(msg.videos || []), toAbsoluteUrl(data.url)],
                isLoading: false
              }
            : msg
        ))
        break

      case 'interpretation':
        // Not displayed
        break

      case 'description':
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, description: data.content, isLoading: false }
            : msg
        ))
        break

      case 'reflection':
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, reflection: data.content }
            : msg
        ))
        break

      case 'enhanced_prompt':
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, enhancedPrompt: data.enhanced, originalPrompt: data.original }
            : msg
        ))
        break

      case 'agent_message':
        // Agent's text response (shown as message content)
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, content: data.content }
            : msg
        ))
        break

      case 'complete':
        // Save session_id for conversation memory
        if (data.session_id) {
          setSessionId(data.session_id)
          console.log('💾 Session ID saved:', data.session_id)
        }
        // Mark generation as complete
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, isLoading: false, isGeneratingImage: false }
            : msg
        ))
        setIsLoading(false)
        break

      case 'error':
        setMessages(prev => prev.map(msg =>
          msg.id === msgId
            ? { ...msg, content: `Error: ${data.message}`, isLoading: false }
            : msg
        ))
        break

      default:
        console.log('Unknown event type:', data.type)
    }
  }

  return (
    <div className="app">
      <div className="chat-container">
        <div className="chat-header">
          <div className="logo">✨</div>
          <h1>Magic Hour AI</h1>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="empty-state" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '3rem', marginBottom: '20px' }}>🎨</div>
              <h2 style={{ color: 'var(--text-secondary)', marginBottom: '12px' }}>Create Something Amazing</h2>
              <p>Ask me to generate images or videos using AI</p>
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <ChatMessage
              key={msg.id || idx}
              message={msg}
              reasoning={msg.role === 'assistant' && msg.isLoading ? currentReasoning : null}
              selectedImage={selectedImage}
              onImageSelect={setSelectedImage}
            />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <form onSubmit={handleSubmit}>
            <div className="input-wrapper">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Describe what you want to create..."
                disabled={isLoading}
              />
              <button type="submit" className="send-button" disabled={isLoading || !input.trim()}>
                {isLoading ? (
                  <>
                    <span className="spinner" style={{ width: 16, height: 16, border: '2px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 1s linear infinite' }}></span>
                    Creating...
                  </>
                ) : (
                  <>Send →</>
                )}
              </button>
            </div>
          </form>

          <div className="settings-toggle" onClick={() => setShowSettings(!showSettings)}>
            ⚙️ Settings {showSettings ? '▼' : '▶'}
          </div>

          {showSettings && (
            <div className="settings-panel">
              <div className="setting-item">
                <label>Mode</label>
                <select 
                  value={settings.mode} 
                  onChange={(e) => setSettings({...settings, mode: e.target.value})}
                >
                  <option value="fast">⚡ Fast</option>
                  <option value="pro">✨ Pro (Higher Quality)</option>
                </select>
              </div>
              <div className="setting-item">
                <label>Aspect Ratio</label>
                <select
                  value={settings.aspectRatio}
                  onChange={(e) => setSettings({...settings, aspectRatio: e.target.value})}
                >
                  <option value="square">Square (1:1)</option>
                  <option value="landscape_4_3">Landscape (4:3)</option>
                  <option value="landscape_16_9">Wide (16:9)</option>
                  <option value="portrait_3_4">Portrait (3:4)</option>
                  <option value="portrait_9_16">Tall (9:16)</option>
                </select>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
