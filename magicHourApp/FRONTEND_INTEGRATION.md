# Frontend Integration Guide - Reasoning UI & Session Management

## 1. Session Management (CRITICAL FIX)

### Problem
The frontend was NOT sending `session_id`, causing conversation history to reset on every request.

### Solution

Store session_id in React state and send it with every request:

```jsx
import { useState, useRef } from 'react';

function Chat() {
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);

  const sendMessage = async (userMessage) => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: userMessage,
        settings: { mode: 'fast', aspectRatio: 'square' },
        history: messages,
        session_id: sessionId  // ✅ CRITICAL: Send session_id
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          // ✅ CRITICAL: Save session_id from complete event
          if (data.type === 'complete') {
            setSessionId(data.session_id);
            console.log('Session ID saved:', data.session_id);
          }

          // Handle other events...
        }
      }
    }
  };

  return (/* your UI */);
}
```

**Key Points:**
1. Store `sessionId` in state
2. Send it with EVERY request
3. Update it from the `complete` event
4. Without this, the agent has NO memory!

---

## 2. Reasoning UI Components

### New SSE Event Types

```typescript
type SSEEvent =
  | { type: 'reasoning_step', content: string, collapsible: boolean }
  | { type: 'visual_analysis', content: string, collapsible: boolean }
  | { type: 'reasoning', content: string }  // Simple status
  | { type: 'image_complete', url: string }
  | { type: 'video_complete', url: string }
  | { type: 'agent_message', content: string }
  | { type: 'complete', duration: number, session_id: string };
```

### Collapsible Reasoning Component

```jsx
import { useState } from 'react';
import { ChevronRight, ChevronDown } from 'lucide-react';

function ReasoningStep({ content, collapsible }) {
  const [expanded, setExpanded] = useState(false);

  if (!collapsible) {
    // Simple status message (not collapsible)
    return (
      <div className="reasoning-status">
        {content}
      </div>
    );
  }

  // Collapsible reasoning
  const lines = content.split('\n');
  const firstLine = lines[0];
  const restContent = lines.slice(1).join('\n');

  return (
    <div className="reasoning-container">
      <button
        onClick={() => setExpanded(!expanded)}
        className="reasoning-toggle"
      >
        {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>{firstLine}</span>
      </button>

      {expanded && (
        <div className="reasoning-content">
          <pre>{restContent}</pre>
        </div>
      )}
    </div>
  );
}
```

### CSS Styling

```css
.reasoning-container {
  margin: 8px 0;
  border-left: 3px solid #3b82f6;
  background: #f1f5f9;
  border-radius: 4px;
}

.reasoning-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  width: 100%;
  background: transparent;
  border: none;
  cursor: pointer;
  text-align: left;
  font-size: 14px;
  color: #1e293b;
  transition: background 0.2s;
}

.reasoning-toggle:hover {
  background: rgba(59, 130, 246, 0.1);
}

.reasoning-content {
  padding: 12px;
  border-top: 1px solid #e2e8f0;
  background: white;
}

.reasoning-content pre {
  margin: 0;
  white-space: pre-wrap;
  font-size: 13px;
  line-height: 1.5;
  color: #475569;
}

.reasoning-status {
  padding: 8px 12px;
  margin: 4px 0;
  background: #f8fafc;
  border-radius: 4px;
  font-size: 14px;
  color: #64748b;
}
```

### Complete Message Handler

```jsx
function ChatInterface() {
  const [messages, setMessages] = useState([]);
  const [reasoningSteps, setReasoningSteps] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  const handleSSEEvent = (event) => {
    switch (event.type) {
      case 'reasoning_step':
        // Add to collapsible reasoning list
        setReasoningSteps(prev => [...prev, {
          content: event.content,
          collapsible: event.collapsible,
          id: Date.now()
        }]);
        break;

      case 'visual_analysis':
        // Visual analysis is also collapsible
        setReasoningSteps(prev => [...prev, {
          content: event.content,
          collapsible: true,
          type: 'visual',
          id: Date.now()
        }]);
        break;

      case 'image_complete':
        setMessages(prev => [...prev, {
          type: 'image',
          url: event.url,
          id: Date.now()
        }]);
        break;

      case 'complete':
        // ✅ SAVE SESSION ID
        setSessionId(event.session_id);
        // Clear reasoning steps for next request
        setReasoningSteps([]);
        break;

      default:
        console.log('Unknown event:', event);
    }
  };

  return (
    <div className="chat-container">
      {/* Reasoning Steps Section */}
      {reasoningSteps.length > 0 && (
        <div className="reasoning-section">
          <h3>🧠 Agent Thinking</h3>
          {reasoningSteps.map(step => (
            <ReasoningStep
              key={step.id}
              content={step.content}
              collapsible={step.collapsible}
            />
          ))}
        </div>
      )}

      {/* Messages Section */}
      <div className="messages-section">
        {messages.map(msg => (
          msg.type === 'image' ? (
            <img key={msg.id} src={msg.url} alt="Generated" />
          ) : (
            <div key={msg.id}>{msg.content}</div>
          )
        ))}
      </div>
    </div>
  );
}
```

---

## 3. Visual Flow Example

When user sends "add scorpio" after "generate sub-zero":

```
┌─────────────────────────────────────┐
│ 💬 User: add scorpio                │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ 🧠 Agent Thinking                   │
│ ┌───────────────────────────────┐   │
│ │ ▶ 👁️ Visual Analysis          │   │
│ │   (Click to expand)           │   │
│ └───────────────────────────────┘   │
│ ┌───────────────────────────────┐   │
│ │ ▼ 💭 Reasoning                │   │
│ │   The user wants to add...    │   │
│ │   From visual analysis, I see │   │
│ │   Sub-Zero from Mortal Kombat │   │
│ │   Therefore "scorpio" = ...   │   │
│ └───────────────────────────────┘   │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ ✅ Image generated                  │
│ ✅ Image generated                  │
│ ✅ Image generated                  │
│ ✅ Image generated                  │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│ ▶ ✅ Content Reflection             │
│   Successfully edited 4 images...   │
└─────────────────────────────────────┘
         ↓
[Images displayed]
```

---

## 4. Testing Checklist

### Test Session Persistence
```javascript
// Request 1
POST /api/chat
{
  "message": "generate sub-zero",
  "session_id": null  // First request
}

// Response should include session_id in complete event
// {"type": "complete", "session_id": "session-1234567890"}

// Request 2 - MUST use same session_id
POST /api/chat
{
  "message": "add scorpio",
  "session_id": "session-1234567890"  // ✅ Same session!
}

// Now agent should have access to Sub-Zero images
```

### Test Reasoning Display
1. Send: "generate sub-zero"
   - Should see reasoning step
   - Should see 4 images
   - Should see content reflection

2. Send: "add scorpio"
   - Should see visual analysis (collapsible)
   - Should see reasoning (collapsible)
   - Should reference Sub-Zero from previous request
   - Should EDIT existing images (not generate new)

---

## 5. Common Issues & Fixes

### Issue: Agent has no memory
**Symptom**: Every request acts like the first one
**Fix**: Ensure session_id is being sent with requests
```jsx
// ❌ Wrong
fetch('/api/chat', {
  body: JSON.stringify({ message: text })
});

// ✅ Correct
fetch('/api/chat', {
  body: JSON.stringify({
    message: text,
    session_id: sessionId  // Must include this!
  })
});
```

### Issue: Reasoning not displaying
**Symptom**: No collapsible sections appear
**Fix**: Handle `reasoning_step` and `visual_analysis` events
```jsx
if (data.type === 'reasoning_step' || data.type === 'visual_analysis') {
  setReasoningSteps(prev => [...prev, data]);
}
```

### Issue: Session ID not saving
**Symptom**: Session ID is null on second request
**Fix**: Update state from `complete` event
```jsx
if (data.type === 'complete') {
  setSessionId(data.session_id);
  localStorage.setItem('sessionId', data.session_id); // Optional: persist across page reloads
}
```

---

## 6. localStorage Persistence (Optional)

To maintain session across page reloads:

```jsx
function usePersistedSession() {
  const [sessionId, setSessionId] = useState(() => {
    // Initialize from localStorage
    return localStorage.getItem('magichour_session_id');
  });

  const updateSessionId = (newId) => {
    setSessionId(newId);
    localStorage.setItem('magichour_session_id', newId);
  };

  const clearSession = () => {
    setSessionId(null);
    localStorage.removeItem('magichour_session_id');
  };

  return { sessionId, updateSessionId, clearSession };
}

// Usage
function Chat() {
  const { sessionId, updateSessionId, clearSession } = usePersistedSession();

  // In your complete handler:
  if (data.type === 'complete') {
    updateSessionId(data.session_id);
  }

  // Add a "New Chat" button:
  const handleNewChat = () => {
    clearSession();
    setMessages([]);
  };
}
```

---

## 7. Complete Integration Example

```jsx
import { useState, useRef, useEffect } from 'react';

function MagicHourChat() {
  const [messages, setMessages] = useState([]);
  const [reasoning, setReasoning] = useState([]);
  const [sessionId, setSessionId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async (text) => {
    setIsLoading(true);
    setReasoning([]); // Clear previous reasoning

    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        settings: { mode: 'fast', aspectRatio: 'square' },
        history: messages,
        session_id: sessionId  // ✅ Pass session_id
      })
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6));

          switch (data.type) {
            case 'reasoning_step':
            case 'visual_analysis':
              setReasoning(prev => [...prev, data]);
              break;

            case 'image_complete':
              setMessages(prev => [...prev, {
                type: 'image',
                url: data.url,
                timestamp: Date.now()
              }]);
              break;

            case 'complete':
              setSessionId(data.session_id);  // ✅ Save session_id
              setIsLoading(false);
              break;
          }
        }
      }
    }
  };

  return (
    <div>
      {/* Reasoning Section */}
      {reasoning.length > 0 && (
        <div className="reasoning-panel">
          {reasoning.map((step, idx) => (
            <ReasoningStep
              key={idx}
              content={step.content}
              collapsible={step.collapsible}
            />
          ))}
        </div>
      )}

      {/* Messages */}
      {messages.map(msg => (
        <img key={msg.timestamp} src={msg.url} />
      ))}

      {/* Input */}
      <input
        onKeyPress={(e) => {
          if (e.key === 'Enter') {
            sendMessage(e.target.value);
            e.target.value = '';
          }
        }}
        disabled={isLoading}
      />
    </div>
  );
}
```

---

## Summary

**Must-Have Changes:**
1. ✅ Add `session_id` to state
2. ✅ Send `session_id` with every request
3. ✅ Save `session_id` from `complete` event
4. ✅ Handle `reasoning_step` and `visual_analysis` events
5. ✅ Create collapsible UI components

**Without session_id, the agent will have amnesia!** 🧠
