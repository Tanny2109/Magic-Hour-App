# Fal.ai Agentic Workflow Guide

## Summary of Your Options

### ✅ Option 1: Smolagents (RECOMMENDED)
**Best for:** Full agentic workflow with tool selection, streaming, multimodal output

**Pros:**
- Native fal.ai support via Hugging Face InferenceClient
- LLM automatically selects tools and parameters
- Proper streaming support
- Clean tool abstraction
- Multimodal Gradio chatbot integration

**Cons:**
- Requires Hugging Face API or other LLM provider
- Learning curve for smolagents framework

**Files:** `smolagent_app.py`, `fal_tools.py`

---

### ⚠️ Option 2: Your Custom LLM Workflow (FIXED)
**Best for:** Simple, direct control, using fal-ai/any-llm

**Pros:**
- Uses fal.ai's own LLM (any-llm)
- Full control over logic
- Simpler to understand

**Cons:**
- Manual tool selection logic
- Less robust than agent frameworks
- More maintenance

**Files:** `gradio_app.py`, `falClient.py`

---

### ❌ Option 3: AG2 (AutoGen)
**Not recommended:** No documented fal.ai integration, would require significant custom work

---

## How to Use Each Option

### Using Smolagents App (RECOMMENDED)

1. **Install dependencies:**
```bash
pip install smolagents gradio fal-client requests pillow huggingface_hub
```

2. **Set up API keys:**
```bash
export FAL_KEY="your-fal-key"
export HF_TOKEN="your-huggingface-token"
```

3. **Run the app:**
```bash
python smolagent_app.py
```

**Features:**
- Agent automatically selects fal.ai model based on request
- Streams thinking process
- Displays multiple images inline in chat
- Handles videos and image editing
- Proper multimodal Gradio Chatbot integration

**Example prompts:**
- "Generate 2 images of a sunset using flux-pro in vertical format"
- "Create a 5-second video of ocean waves"
- "Make this image more vibrant and colorful"

---

### Using Fixed Custom App

1. **Install dependencies:**
```bash
pip install gradio fal-client requests pillow
```

2. **Set up API key:**
```bash
export FAL_KEY="your-fal-key"
```

3. **Run the app:**
```bash
python gradio_app.py
```

**What was fixed:**
- `interact_with_agent()` now properly yields message dictionaries with `role` and `content`
- Content format supports both text and multiple images
- Properly handles `type="messages"` format for gr.Chatbot
- Filters out error messages before adding as images

**Message format example:**
```python
{
    "role": "assistant",
    "content": [
        {"type": "text", "text": "Here are your images!"},
        {"type": "image", "image": "/path/to/image1.png"},
        {"type": "image", "image": "/path/to/image2.png"}
    ]
}
```

---

## Key Differences in Implementation

### Gradio Chatbot with `type="messages"`

**Correct format:**
```python
# History is a list of message dictionaries
history = [
    {"role": "user", "content": "Generate an image"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Here you go!"},
            {"type": "image", "image": "/path/to/image.png"}
        ]
    }
]

chatbot = gr.Chatbot(type="messages")
```

**For streaming:**
```python
def interact(prompt, history):
    history = history or []
    history.append({"role": "user", "content": prompt})
    yield history

    # Do processing...

    history.append({
        "role": "assistant",
        "content": [
            {"type": "text", "text": response_text},
            {"type": "image", "image": image_path}
        ]
    })
    yield history
```

---

## Fal.ai Model Selection Logic

### In Smolagents:
The LLM agent automatically selects the appropriate tool and parameters based on natural language.

**Example:**
- User: "Generate 2 vertical images of a dragon using flux-pro"
- Agent: Calls `FalImageGenerationTool(prompt="dragon", model="flux-pro", width=512, height=768, num_images=2)`

### In Custom App:
Your LLM (fal-ai/any-llm) returns JSON with operation and parameters, which you parse and execute.

**Example:**
```json
{
  "operation": "generate_image",
  "model": "flux-pro",
  "prompt": "dragon",
  "parameters": {
    "width": 512,
    "height": 768,
    "num_images": 2
  }
}
```

---

## Streaming Support

### Smolagents:
```python
def stream_agent_response(self, user_message, history):
    history.append({"role": "user", "content": user_message})
    yield history

    # Show thinking
    history.append({"role": "assistant", "content": "🤔 Thinking..."})
    yield history

    # Get result
    result = self.agent.run(user_message)

    # Update with result + images
    history[-1]["content"] = [
        {"type": "text", "text": result},
        {"type": "image", "image": image_path}
    ]
    yield history
```

### Custom App (Fixed):
```python
def interact_with_agent(self, prompt, history):
    history = history or []
    history.append({"role": "user", "content": prompt})
    yield history

    imgs, text = self.assistant.process_message(prompt, history)

    content = [{"type": "text", "text": text}]
    for img in imgs:
        content.append({"type": "image", "image": img})

    history.append({"role": "assistant", "content": content})
    yield history
```

---

## Available Fal.ai Models

### Image Generation:
- `nano-banana` - Fast, high quality
- `flux-schnell` - Very fast
- `flux-pro` - Highest quality (slower)

### Video Generation:
- `luma-dream-machine` - Text-to-video

### Image Editing:
- `flux/dev/image-to-image` - Image transformation
- `flux-inpainting` - Image editing with masks

---

## Troubleshooting

### Issue: "OSError: File name too long"
**Cause:** Passing error messages as image paths to `gr.Image()`

**Fix:** Filter errors before creating image components
```python
if isinstance(img, str) and img.startswith("Error"):
    continue  # Skip errors
```

### Issue: Chatbot not displaying images
**Cause:** Wrong message format or missing `type="messages"`

**Fix:**
1. Use `gr.Chatbot(type="messages")`
2. Use proper content structure: `{"type": "image", "image": path}`

### Issue: Agent not streaming
**Cause:** Not yielding history after each update

**Fix:** Yield history dictionary after each change
```python
yield history  # Must yield to stream!
```

---

## Next Steps

1. **Try the smolagents app first** - It's more robust and requires less manual logic
2. **For custom control**, use the fixed `gradio_app.py`
3. **Extend tools** by adding more fal.ai models to `fal_tools.py`
4. **Improve streaming** by adding progress indicators during image generation

---

## Example Usage

### Smolagents App:
```
User: "Create 2 images of a futuristic city, make them wide format"

Agent: 🤔 Thinking...
       I'll use the fal_image_generation tool with:
       - prompt: "futuristic city"
       - width: 1024
       - height: 576
       - num_images: 2

       [Displays 2 images inline]
```

### Custom App:
```
User: "Generate an image of a sunset"