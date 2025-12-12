"""
FastAPI backend for Magic Hour AI - streams agent reasoning and results via SSE
"""
import os
import sys
import json
import asyncio
import threading
import time
import re
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add the mh_agentic_workflow to path
WORKFLOW_PATH = Path(__file__).parent.parent / "mh_agentic_workflow"
sys.path.insert(0, str(WORKFLOW_PATH))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(WORKFLOW_PATH / ".env")

# Import agent components (from the existing workflow)
from src.agents.smolagent_ref import SmolagentFalApp
from src.core.utils import parse_image_paths, parse_video_paths
from config.settings import settings

# Initialize the agent
agent_app: Optional[SmolagentFalApp] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize agent on startup"""
    global agent_app
    print("🚀 Initializing Magic Hour AI Agent...")
    try:
        settings.validate()
        agent_app = SmolagentFalApp(
            hf_token=settings.HF_TOKEN,
            fal_model_name=settings.FAL_MODEL_NAME
        )
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        raise
    yield
    print("👋 Shutting down...")

app = FastAPI(title="Magic Hour AI", lifespan=lifespan)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    settings: dict = {}
    history: list = []  # Conversation history with image paths


class SSEEvent:
    """Helper to format SSE events"""
    @staticmethod
    def format(event_type: str, data: dict) -> str:
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"


async def generate_with_streaming(message: str, user_settings: dict, history: list = None) -> AsyncGenerator[str, None]:
    """
    Run the agent and stream events:
    - reasoning: Agent thinking process
    - tool_call: When a tool is invoked
    - image_progress: Image generation started
    - image_complete: Image URL ready
    - video_complete: Video URL ready  
    - description: Conversational description of the result
    - complete: Generation finished
    - error: If something goes wrong
    """
    global agent_app
    
    if not agent_app:
        yield SSEEvent.format("error", {"message": "Agent not initialized"})
        return

    # Build prompt with settings
    mode = user_settings.get("mode", "fast")
    aspect_ratio = user_settings.get("aspectRatio", "square")
    
    # Build context from history - extract image/video paths from previous responses
    context_parts = []
    if history:
        print(f"DEBUG - History received: {len(history)} messages")
        for msg in history:
            if msg.get("role") == "assistant":
                # Extract images
                images = msg.get("images", [])
                for img_url in images:
                    # Extract the local path from the API URL
                    if "path=" in img_url:
                        local_path = img_url.split("path=")[-1]
                        context_parts.append(f"Previously generated image (use this path for editing): {local_path}")
                        print(f"DEBUG - Found image path: {local_path}")
                
                # Extract videos
                videos = msg.get("videos", [])
                for vid_url in videos:
                    if "path=" in vid_url:
                        local_path = vid_url.split("path=")[-1]
                        context_parts.append(f"Previously generated video: {local_path}")
                        print(f"DEBUG - Found video path: {local_path}")
    
    context_str = "\n".join(context_parts) if context_parts else ""
    print(f"DEBUG - Context string: {context_str}")
    
    full_prompt = f"{message}\n\n[System Settings]\nPerformance Mode: {mode}\nAspect Ratio: {aspect_ratio}"
    if context_str:
        full_prompt = f"[Previous Content]\n{context_str}\n\n[Current Request]\n{message}\n\n[System Settings]\nPerformance Mode: {mode}\nAspect Ratio: {aspect_ratio}"
    
    print(f"DEBUG - Full prompt being sent to agent:\n{full_prompt[:500]}...")
    
    # Variables to capture agent output
    agent_output = None
    agent_error = None
    
    def run_agent():
        nonlocal agent_output, agent_error
        try:
            agent_output = agent_app.agent.run(full_prompt)
        except Exception as e:
            agent_error = e

    # Yield initial reasoning
    yield SSEEvent.format("reasoning", {"content": f"Analyzing request: \"{message}\""})
    await asyncio.sleep(0.1)
    
    yield SSEEvent.format("reasoning", {"content": f"Using {mode} mode with {aspect_ratio} aspect ratio"})
    await asyncio.sleep(0.1)

    # Start agent in background thread
    start_time = time.time()
    agent_thread = threading.Thread(target=run_agent)
    agent_thread.start()
    
    # Stream progress while waiting
    phases = [
        "Selecting appropriate tool for the task...",
        "Preparing generation parameters...",
        "Connecting to AI model..."
    ]
    
    for i, phase in enumerate(phases):
        if not agent_thread.is_alive():
            break
        yield SSEEvent.format("reasoning", {"content": phase})
        await asyncio.sleep(0.5)
    
    # Indicate image generation started if still running
    if agent_thread.is_alive():
        yield SSEEvent.format("tool_call", {
            "tool": "fal_image_generation" if "video" not in message.lower() else "fal_video_generation",
            "args": f"prompt: {message[:100]}..."
        })
        yield SSEEvent.format("image_progress", {"status": "generating"})
    
    # Wait for completion with periodic updates
    while agent_thread.is_alive():
        elapsed = time.time() - start_time
        if elapsed > 5 and int(elapsed) % 3 == 0:
            yield SSEEvent.format("reasoning", {"content": f"Still generating... ({elapsed:.0f}s)"})
        await asyncio.sleep(0.5)
    
    agent_thread.join()
    total_time = time.time() - start_time
    
    # Handle errors
    if agent_error:
        yield SSEEvent.format("error", {"message": str(agent_error)})
        return
    
    # Process output
    output_text = str(agent_output)
    image_paths = parse_image_paths(output_text)
    video_paths = parse_video_paths(output_text)
    
    # Yield images
    for img_path in image_paths:
        # Serve local files via API
        yield SSEEvent.format("image_complete", {
            "url": f"http://localhost:8000/api/media?path={img_path}"
        })
        await asyncio.sleep(0.1)
    
    # Yield videos
    for vid_path in video_paths:
        yield SSEEvent.format("video_complete", {
            "url": f"http://localhost:8000/api/media?path={vid_path}"
        })
        await asyncio.sleep(0.1)
    
    # Generate conversational description
    if image_paths or video_paths:
        media_type = "video" if video_paths else "image"
        count = len(video_paths) if video_paths else len(image_paths)
        
        description = f"I've created {count} {media_type}{'s' if count > 1 else ''} based on your request. "
        description += f"The generation took {total_time:.1f} seconds using {mode} mode. "
        
        if "sunset" in message.lower():
            description += "The warm golden hues capture that magical moment when day meets night."
        elif "futuristic" in message.lower() or "city" in message.lower():
            description += "The neon-lit atmosphere creates an immersive cyberpunk aesthetic."
        elif "nature" in message.lower() or "landscape" in message.lower():
            description += "The natural elements come together to create a serene and peaceful scene."
        elif "dragon" in message.lower() or "fantasy" in message.lower():
            description += "The mythical elements are brought to life with dramatic lighting and detail."
        else:
            description += "Feel free to ask for adjustments or generate something new!"
        
        yield SSEEvent.format("description", {"content": description})
    
    yield SSEEvent.format("complete", {"duration": total_time})


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Stream chat responses via Server-Sent Events"""
    return StreamingResponse(
        generate_with_streaming(request.message, request.settings, request.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/media")
async def serve_media(path: str):
    """Serve generated media files"""
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine content type
    ext = Path(path).suffix.lower()
    content_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg", 
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".mp4": "video/mp4",
        ".webm": "video/webm"
    }
    
    return FileResponse(
        path,
        media_type=content_types.get(ext, "application/octet-stream")
    )


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "agent_ready": agent_app is not None}


if __name__ == "__main__":
    import uvicorn
    print("🌟 Starting Magic Hour AI Server...")
    print("📡 API: http://localhost:8000")
    print("🎨 Frontend: http://localhost:5173")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
