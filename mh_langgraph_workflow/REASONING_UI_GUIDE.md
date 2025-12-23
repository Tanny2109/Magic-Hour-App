# Reasoning & Enhanced Prompts Implementation Guide

## Overview

The LangGraph agent now supports:
1. **Collapsible Reasoning UI** - Step-by-step thinking displayed with dropdown
2. **Enhanced Prompts** - Settings, visual analysis, and image paths
3. **Content Reflection** - Summary of what was generated

## Custom Message Types

### 1. `ReasoningMessage`
Contains agent's reasoning/thinking process.

**Example:**
```
[REASONING]
The user wants to add scorpio. From [VISUAL CONTEXT ANALYSIS], I can see the previous images show Sub-Zero from Mortal Kombat. Therefore "scorpio" likely refers to Scorpion, another MK character. I should EDIT the most recent Sub-Zero images to add Scorpion in the same style.
```

### 2. `VisualAnalysisMessage`
Contains vision model's analysis of previous images.

**Example:**
```
👁️ Analyzing images from conversation...

Visual Analysis:
I see 4 images showing Sub-Zero from Mortal Kombat in a fighting stance with ice-blue ninja outfit. The art style is realistic digital art with dramatic lighting. Given the user's request "add scorpio", this likely refers to adding Scorpion (the yellow/orange ninja rival) to maintain the Mortal Kombat theme. I recommend EDITING the recent Sub-Zero images to add Scorpion in the same realistic fighting game style.
```

### 3. Content Reflection
After generation, agent reflects on what was created.

**Example:**
```
✅ Successfully generated 4 image(s) with the prompt: 'Sub-Zero and Scorpion from Mortal Kombat in fighting poses'
```

## SSE Event Types

The API server sends these events for UI rendering:

### `reasoning_step`
```json
{
  "type": "reasoning_step",
  "content": "The user wants to add scorpio...",
  "collapsible": true
}
```

### `visual_analysis`
```json
{
  "type": "visual_analysis",
  "content": "I see 4 images showing Sub-Zero...",
  "collapsible": true
}
```

### `reasoning` (legacy - for simple status updates)
```json
{
  "type": "reasoning",
  "content": "✅ Image generated"
}
```

## UI Implementation

### Collapsible Reasoning Component

The UI should render `reasoning_step` and `visual_analysis` events with a dropdown/accordion:

```jsx
// Example React component
function ReasoningStep({ content, collapsible }) {
  const [expanded, setExpanded] = useState(false);

  if (!collapsible) {
    return <div className="status-update">{content}</div>;
  }

  return (
    <div className="reasoning-container">
      <button
        onClick={() => setExpanded(!expanded)}
        className="reasoning-toggle"
      >
        {expanded ? '▼' : '▶'} {content.split('\n')[0]}
      </button>
      {expanded && (
        <div className="reasoning-content">
          {content}
        </div>
      )}
    </div>
  );
}
```

### Visual Flow Example

```
User: "add scorpio"
↓
[Collapsible] 👁️ Analyzing images from conversation...
  └─ (Click to expand) Visual Analysis: I see Sub-Zero from Mortal Kombat...
↓
[Collapsible] 💭 Reasoning
  └─ (Click to expand) The user wants to add scorpio. From visual analysis...
↓
✅ Image generated (status - not collapsible)
↓
[Collapsible] ✅ Content Reflection
  └─ Successfully generated 4 image(s) with prompt: '...'
↓
[Images displayed]
```

## Enhanced Prompt Structure

The agent builds prompts with this structure:

```
User request: add scorpio

[Settings]
Mode: fast | Aspect Ratio: square
Default: Generate 4 variations unless user specifies otherwise.

[VISUAL CONTEXT ANALYSIS]
I see 4 images showing Sub-Zero from Mortal Kombat in a fighting stance...

[Image File Paths]
  - Batch 3 Image: /tmp/tmpxxx1.png
  - Batch 3 Image: /tmp/tmpxxx2.png
  - Batch 3 Image: /tmp/tmpxxx3.png
  - Batch 3 Image: /tmp/tmpxxx4.png

Based on the visual analysis above, execute the user's request appropriately.
```

## Testing

Test these scenarios:

1. **Fresh Generation** (no history)
   - Should NOT run visual analysis
   - Shows settings only

2. **Edit Request** (`"add X"` after previous generation)
   - SHOULD run visual analysis
   - Shows collapsible visual analysis
   - Shows collapsible reasoning
   - Shows content reflection after completion

3. **Ambiguous Reference** (`"add scorpio"` after Sub-Zero)
   - Visual analysis identifies Mortal Kombat theme
   - Reasoning interprets "scorpio" as Scorpion character
   - Edits only recent batch (not all history)

## Benefits

✅ **Smarter Context Understanding** - Vision analysis prevents wrong interpretations
✅ **Transparency** - Users see the agent's thought process
✅ **Performance** - Visual analysis only runs when needed
✅ **Better UX** - Collapsible UI keeps interface clean
✅ **Traceability** - Each step is logged and visible
