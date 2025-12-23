"""FastAPI server for the LangGraph content generation agent."""
import os
import json
import asyncio
import tempfile
import base64
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from src.agents import create_agent

# Initialize FastAPI app
app = FastAPI(
    title="Magic Hour LangGraph API",
    description="Content generation API using LangChain and LangGraph",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (per-thread isolation via thread_id)
agent = create_agent(
    fal_model_name=os.getenv("FAL_MODEL_NAME", "google/gemini-2.5-flash"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
    max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
)


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str  # "user" or "assistant"
    content: str
    image_paths: Optional[List[str]] = None


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    message: str
    image_paths: Optional[List[str]] = None  # Paths to images to include
    image_data: Optional[List[str]] = None  # Base64 encoded images
    thread_id: str = "default"
    settings: Optional[dict] = None  # mode, aspect_ratio, etc.


class ChatResponse(BaseModel):
    """Response body for chat endpoint."""
    message: str
    generated_content: List[str] = []
    thread_id: str


def _save_base64_image(base64_data: str) -> str:
    """Save a base64 encoded image to a temp file and return the path."""
    # Handle data URL format
    if "," in base64_data:
        base64_data = base64_data.split(",")[1]

    image_bytes = base64.b64decode(base64_data)

    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temp_file.write(image_bytes)
    temp_file.close()

    return temp_file.name


def _extract_response_text(messages) -> str:
    """Extract the final response text from agent messages."""
    from langchain_core.messages import AIMessage

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                return msg.content
            elif isinstance(msg.content, list):
                # Handle multimodal responses
                text_parts = [p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
                return " ".join(text_parts)

    return ""


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "langgraph-agent"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message and return the response."""
    # Prepare image paths
    image_paths = request.image_paths or []

    # Save any base64 images to temp files
    if request.image_data:
        for b64_data in request.image_data:
            try:
                path = _save_base64_image(b64_data)
                image_paths.append(path)
            except Exception as e:
                print(f"Failed to save base64 image: {e}")

    # Invoke the agent
    try:
        result = agent.invoke(
            message=request.message,
            image_paths=image_paths if image_paths else None,
            thread_id=request.thread_id,
            settings=request.settings
        )

        response_text = _extract_response_text(result.get("messages", []))
        generated_content = result.get("generated_content", [])

        return ChatResponse(
            message=response_text,
            generated_content=generated_content,
            thread_id=request.thread_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream the agent's response as Server-Sent Events."""

    async def event_generator():
        # Prepare image paths
        image_paths = request.image_paths or []

        # Save any base64 images to temp files
        if request.image_data:
            for b64_data in request.image_data:
                try:
                    path = _save_base64_image(b64_data)
                    image_paths.append(path)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'content': f'Failed to save image: {e}'})}\n\n"

        try:
            # Stream agent responses
            for event in agent.stream(
                message=request.message,
                image_paths=image_paths if image_paths else None,
                thread_id=request.thread_id,
                settings=request.settings
            ):
                messages = event.get("messages", [])
                generated = event.get("generated_content", [])

                # Process messages
                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        # Tool call event
                        for tool_call in msg.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'name': tool_call.get('name', 'unknown'), 'args': tool_call.get('args', {})})}\n\n"

                    elif hasattr(msg, "content"):
                        # Check message type
                        msg_type = type(msg).__name__

                        if msg_type == "ToolMessage":
                            # Tool result
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            yield f"data: {json.dumps({'type': 'tool_result', 'content': content})}\n\n"

                            # Check for generated content paths
                            import re
                            paths = re.findall(r'(/[^\s]+\.(?:png|jpg|jpeg|webp|mp4))', content)
                            for path in paths:
                                if os.path.exists(path):
                                    if path.endswith(".mp4"):
                                        yield f"data: {json.dumps({'type': 'video_complete', 'path': path})}\n\n"
                                    else:
                                        yield f"data: {json.dumps({'type': 'image_complete', 'path': path})}\n\n"

                        elif msg_type == "AIMessage":
                            content = msg.content if isinstance(msg.content, str) else str(msg.content)
                            yield f"data: {json.dumps({'type': 'assistant', 'content': content})}\n\n"

                # Allow other tasks to run
                await asyncio.sleep(0.01)

            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'generated_content': generated})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/media")
async def get_media(path: str):
    """Serve generated media files."""
    # Security: Validate path is within temp directory
    temp_dir = tempfile.gettempdir()
    resolved_path = Path(path).resolve()

    if not str(resolved_path).startswith(temp_dir):
        raise HTTPException(status_code=403, detail="Access denied")

    if not resolved_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    suffix = resolved_path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".mp4": "video/mp4",
        ".webm": "video/webm",
    }

    media_type = media_types.get(suffix, "application/octet-stream")

    return FileResponse(
        path=str(resolved_path),
        media_type=media_type,
        filename=resolved_path.name
    )


@app.get("/history/{thread_id}")
async def get_history(thread_id: str):
    """Get conversation history for a thread."""
    messages = agent.get_conversation_history(thread_id)
    generated = agent.get_generated_content(thread_id)

    # Convert messages to serializable format
    history = []
    for msg in messages:
        msg_type = type(msg).__name__
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        history.append({
            "type": msg_type,
            "content": content
        })

    return {
        "thread_id": thread_id,
        "messages": history,
        "generated_content": generated
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True
    )
