# Architecture Overview

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface (Gradio)                   │
│                     Port 7860 - Web Browser                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ├─ main_smolagent.py (Recommended)
                                └─ main_legacy.py    (Legacy)
                                │
┌───────────────────────────────┴─────────────────────────────────┐
│                         Application Layer                        │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  src/agents/                                             │  │
│  │  ┌──────────────────┐     ┌──────────────────┐          │  │
│  │  │ smolagent.py     │     │ legacy_agent.py  │          │  │
│  │  │                  │     │                  │          │  │
│  │  │ - Agent Logic    │     │ - Intent Parse   │          │  │
│  │  │ - Tool Selection │     │ - Manual Routing │          │  │
│  │  │ - Streaming      │     │ - Simple Flow    │          │  │
│  │  └──────────────────┘     └──────────────────┘          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                │                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  src/tools/                                              │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │ FalImageGenerationTool                             │  │  │
│  │  │ FalVideoGenerationTool                             │  │  │
│  │  │ FalImageEditTool                                   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                │                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  src/core/                                               │  │
│  │  - parse_image_paths()                                   │  │
│  │  - stream_from_smolagent()                               │  │
│  │  - pull_message_from_step()                              │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────┴───────────────────────────────────┐
│                        External Services                         │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │   Fal.ai API    │    │  Hugging Face    │                   │
│  │                 │    │                  │                   │
│  │ - Image Gen     │    │ - LLM (Llama)    │                   │
│  │ - Video Gen     │    │ - Inference      │                   │
│  │ - Image Edit    │    │                  │                   │
│  └─────────────────┘    └──────────────────┘                   │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow: Smolagents Version

```
1. User Input
   │
   ├─ "Generate 2 images of a sunset"
   │
   ▼
2. Agent (smolagent.py)
   │
   ├─ Receives prompt
   ├─ Analyzes request with LLM
   ├─ Selects: FalImageGenerationTool
   ├─ Determines parameters:
   │  - prompt: "sunset"
   │  - num_images: 2
   │  - model: "nano-banana"
   │
   ▼
3. Tool Execution (fal_tools.py)
   │
   ├─ FalImageGenerationTool.forward()
   ├─ Calls Fal.ai API
   ├─ Downloads images
   ├─ Saves to temp files
   │
   ▼
4. Response Streaming (utils.py)
   │
   ├─ Stream thinking: "Analyzing request..."
   ├─ Stream tool call: "Using fal_image_generation"
   ├─ Stream result: "Generated 2 images"
   ├─ Display images inline
   │
   ▼
5. UI Display (Gradio)
   │
   └─ Shows conversation + images
```

## Module Dependency Graph

```
main_smolagent.py
    │
    ├─→ config.settings
    │   └─→ python-dotenv
    │
    └─→ src.agents.smolagent
        ├─→ smolagents (CodeAgent, Model)
        ├─→ gradio
        │
        └─→ src.tools.fal_tools
            ├─→ fal_client
            ├─→ requests
            ├─→ PIL
            └─→ smolagents (Tool)
```

## Request Flow Comparison

### Smolagents (Recommended)
```
User → Agent → LLM decides tool → Tool executes → Stream results
         ↓
    Auto-selection
    Full streaming
    Multimodal output
```

### Legacy
```
User → LLM intent → Manual routing → Tool executes → Return results
         ↓
    Parse JSON response
    Manual tool mapping
    Simple output
```

## Component Responsibilities

### src/agents/
**Purpose**: High-level orchestration and decision making
- Agent logic and workflow
- Tool selection and coordination
- Response streaming
- Error handling

### src/tools/
**Purpose**: Specific capabilities and API interactions
- Fal.ai API calls
- Image/video generation
- File handling
- Result formatting

### src/ui/
**Purpose**: User interaction and presentation
- Gradio interface setup
- Input handling
- Output formatting
- Example prompts

### src/core/
**Purpose**: Shared utilities and helpers
- Path parsing
- Message formatting
- Stream utilities
- Common functions

### config/
**Purpose**: Application configuration
- Environment variables
- Default settings
- Validation
- Constants

## State Management

```
┌─────────────────────────┐
│   Stateless Design      │
├─────────────────────────┤
│ Agent:                  │
│   - Recreated per run   │
│   - No persistent state │
│                         │
│ History:                │
│   - Passed in messages  │
│   - Managed by Gradio   │
│                         │
│ Files:                  │
│   - Temp files          │
│   - Cleaned by OS       │
└─────────────────────────┘
```

## Error Handling Flow

```
User Input
    │
    ▼
Try: Agent.run()
    │
    ├─ Success → Format response → Display
    │
    └─ Exception
        │
        ├─ Log error
        ├─ Format user-friendly message
        └─ Display error in chat
```

## Configuration Loading

```
.env file
    │
    ├─ FAL_KEY=xxx
    └─ HF_TOKEN=xxx
    │
    ▼
python-dotenv loads
    │
    ▼
config/settings.py
    │
    ├─ Settings class
    ├─ Default values
    └─ Validation
    │
    ▼
Used throughout app
```

## Testing Structure

```
tests/
├── test_tools.py
│   └── Mock fal_client
│       └── Test tool logic
│
├── test_utils.py
│   └── Test pure functions
│       └── No mocking needed
│
└── test_config.py
    └── Mock environment
        └── Test validation
```

## Deployment Options

```
Development:
    python main_smolagent.py
    └─→ Gradio dev server (port 7860)

Production:
    Option 1: Docker
        └─→ Container with all dependencies
    
    Option 2: Cloud
        └─→ Hugging Face Spaces
        └─→ AWS/GCP/Azure
    
    Option 3: Server
        └─→ nginx + gunicorn
        └─→ systemd service
```

## Security Layers

```
1. Environment Variables
   └─ API keys not in code

2. .gitignore
   └─ Sensitive files excluded

3. Input Validation
   └─ Sanitize user input

4. Rate Limiting (TODO)
   └─ Prevent abuse

5. Authentication (TODO)
   └─ User access control
```

## Performance Considerations

```
Bottlenecks:
1. Fal.ai API calls (seconds)
   └─ Solution: Show streaming progress

2. Image downloads (network)
   └─ Solution: Parallel downloads

3. LLM inference (seconds)
   └─ Solution: Stream thinking

Optimizations:
- Cache results (future)
- Batch requests (future)
- Optimize prompts
- Smaller models for simple tasks
```

---

**Key Takeaway**: The new structure separates concerns into logical modules, making it easy to understand, test, and extend each component independently.
