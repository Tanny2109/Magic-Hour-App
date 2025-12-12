import { useState, useRef, useEffect } from 'react'
import './App.css'
import ChatMessage from './components/ChatMessage'
import ReasoningPanel from './components/ReasoningPanel'

const API_URL = 'http://localhost:8000'

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

      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          settings: settings,
          history: historyForContext
        })
      })

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
        setCurrentReasoning(prev => [...prev, { 
          type: 'thought', 
          content: data.content,
          timestamp: Date.now()
        }])
        break

      case 'tool_call':
        setCurrentReasoning(prev => [...prev, { 
          type: 'tool', 
          name: data.tool,
          args: data.args,
          timestamp: Date.now()
        }])
        break

      case 'image_progress':
        setMessages(prev => prev.map(msg => 
          msg.id === msgId 
            ? { ...msg, isGeneratingImage: true }
            : msg
        ))
        break

      case 'image_complete':
        setMessages(prev => prev.map(msg => 
          msg.id === msgId 
            ? { 
                ...msg, 
                images: [...(msg.images || []), data.url],
                isGeneratingImage: false,
                isLoading: false
              }
            : msg
        ))
        break

      case 'video_complete':
        setMessages(prev => prev.map(msg => 
          msg.id === msgId 
            ? { 
                ...msg, 
                videos: [...(msg.videos || []), data.url],
                isLoading: false
              }
            : msg
        ))
        break

      case 'description':
        setMessages(prev => prev.map(msg => 
          msg.id === msgId 
            ? { ...msg, description: data.content, isLoading: false }
            : msg
        ))
        break

      case 'complete':
        setMessages(prev => prev.map(msg => 
          msg.id === msgId 
            ? { ...msg, isLoading: false }
            : msg
        ))
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
